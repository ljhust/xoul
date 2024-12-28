from typing import Dict, Optional, List
import logging
import argparse

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

from dataclasses import dataclass
from typing import Optional
import yaml

logger = logging.getLogger("ray.serve")

app = FastAPI()

# chat template for transformer>=4.5 is inevitable
DEFAULT_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}
{% if system_message %}{{ system_message }}
{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""

def load_yaml_config(yaml_path: str) -> Dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"can't find file: {yaml_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML wrong format: {str(e)}")

model_config = load_yaml_config("model.yaml")

@dataclass
class LoRAModulePath:
    path: str
    adapter_name: Optional[str] = None

# have to transform lora pairs into LoRAModulePath type
def parse_lora_modules(lora_paths: str) -> Optional[List[LoRAModulePath]]:
    """Parse comma-separated LoRA paths into LoRAModulePath objects"""
    if not lora_paths:
        return None
    
    lora_modules = []
    for path in lora_paths.split(','):
        path = path.strip()
        if ':' in path:
            # Adapter name is before the colon, path is after
            adapter_name, path = path.split(':', 1)
            lora_modules.append(LoRAModulePath(path=path, adapter_name=adapter_name))
        else:
            # Otherwise, only use the path
            lora_modules.append(LoRAModulePath(path=path))
    
    return lora_modules

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,  # Minimum number of replicas to maintain
        "max_replicas": 10,  # Maximum number of replicas that can be created
        "target_ongoing_requests": 5,  # Target number of ongoing requests per replica
    },
    max_ongoing_requests=10,  # Maximum number of ongoing requests allowed per replica
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = model_config["lora_modules"],
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.base_serving_chat = None
        self.lora_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.base_model = model_config['base_model']
        self.lora_model = model_config['lora_model']

        self.lora_modules = parse_lora_modules(self.lora_modules)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        # Decide which service to use based on the requested model name
        is_lora = request.model == self.lora_model

        if is_lora:
            if not self.lora_serving_chat:
                model_config = await self.engine.get_model_config()
                self.lora_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    [self.lora_model],
                    self.response_role,
                    lora_modules=self.lora_modules,
                    prompt_adapters=self.prompt_adapters,
                    request_logger=self.request_logger,
                    chat_template=self.chat_template,
                )
            return await self.lora_serving_chat.create_chat_completion(request, raw_request)
        else:
            if not self.base_serving_chat:
                model_config = await self.engine.get_model_config()
                self.base_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    [self.base_model],
                    self.response_role,
                    lora_modules=None,  # Base model does not use LoRA
                    prompt_adapters=self.prompt_adapters,
                    request_logger=self.request_logger,
                    chat_template=self.chat_template,
                )
            return await self.base_serving_chat.create_chat_completion(request, raw_request)


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


# Modify the build_app function to require only one deployment
def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds a single Serve app that can handle both base and LoRA models."""
    if "accelerator" in cli_args.keys():
        accelerator = cli_args.pop("accelerator")
    else:
        accelerator = "GPU"
    
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, accelerator: 1})  # for the vLLM actors

    # Create a single deployment that supports both base and LoRA models
    deployment = VLLMDeployment.options(
        name="model_server",
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )

    return deployment