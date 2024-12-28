import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, \
                        get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.notebook import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import glob
from collections import OrderedDict
import re
import wandb
import os

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to load configuration from a YAML file
def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config("config.yaml")

# Set up logging
logger = get_logger(__name__, log_level='INFO')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.logger.addHandler(stream_handler)

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project=config["wandb"]["project"],
    config=config["training"]  # Sync training config with wandb
)

# Load model and tokenizer
training_config = config["training"]
tokenizer = AutoTokenizer.from_pretrained(training_config["model_name"])

# Load and preprocess dataset
try:
    data = load_dataset(
        "text",
        data_files={"train": training_config["train_data_path"]}
    )
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Tokenize dataset
data = data.map(
    lambda samples: tokenizer(
        samples["text"], 
        max_length=training_config["max_length"],  # Use max_length from config
        truncation=True
    ), 
    batched=True
)

# Function to collate data into batches
def collate_fn(batch):
    inputs = [torch.tensor(b['input_ids']) for b in batch]
    input_ids = pad_sequence(inputs, batch_first=True, 
                             padding_value=tokenizer.pad_token_id)
    # -100 is the default ignoring index of CrossEntropyLoss
    labels = pad_sequence(inputs, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'labels': labels}

# Function to delete past model checkpoints
def del_past_models(save_path, file_exten='pth'):
    past_models = glob.glob(os.path.join(save_path, '*.' + file_exten))
    for past_model in past_models:
        os.remove(past_model)
        logger.info(f'Remove model {past_model}!')

# Function to save model checkpoints
def save_checkpoint(path, model, optim, sched, epoch, iters):
    try:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists
        checkpoint_path = os.path.join(path, f'checkpoint_{iters + 1}.pth')
        lr = optim.param_groups[0]['lr']
        model_state = OrderedDict((name, param) for name, param in model.named_parameters() if param.requires_grad)
        logger.info(f"Model of epoch {epoch} saved at checkpoint_{iters + 1}.pth, lr={lr:.3e}")
        torch.save({
            'model': model_state,
            'optimizer': optim.state_dict(),
            'scheduler': sched.state_dict(), 
            'epoch': epoch
        }, checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise

# Function to load model checkpoints
def load_checkpoint(checkpoint_path, model, optim=None, sched=None):
    checkpoint = torch.load(checkpoint_path)
    logger.info(f"Model of epoch {checkpoint['epoch']} is loaded")
    model.load_state_dict(checkpoint['model'], strict=False)
    if optim is not None and sched is not None:
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        return model, optim, sched, checkpoint['epoch']
    else:
        return model, checkpoint['epoch']

# Function to train the model for one epoch
def train_epoch(epoch, model, accelerator, train_dataloader, checkpointing_steps, 
                optimizer, lr_scheduler, save_path):
    global overall_step
    model.train()
    epoch_loss = []
    pbar = tqdm(train_dataloader)
    grad = torch.tensor(0.0)

    for step, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad = accelerator.clip_grad_norm_(parameters=model.parameters(), max_norm=2.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        overall_step += 1
        lr = optimizer.param_groups[0]['lr']

        pbar.set_description(
            f'Epoch {epoch}: loss = {loss.item(): .3f}, grad = {grad.item(): .3f}, lr = {lr: .3e}')
        
        with torch.no_grad():
            avg_loss = accelerator.gather(loss.repeat(len(batch))).mean()
        epoch_loss.append(avg_loss.item() / accelerator.gradient_accumulation_steps)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "step": overall_step,
            "loss": avg_loss.item(),
            "learning_rate": lr,
            "gradient_norm": grad.item(),
        })
        
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            if overall_step % checkpointing_steps == 0:
                del_past_models(save_path)
                output_dir = f"step_{overall_step}.pth"
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(save_path, unwrapped_model, optimizer, lr_scheduler, epoch, overall_step)

    logger.info(f'Epoch {epoch}: loss = {sum(epoch_loss) / len(epoch_loss): .3f}, lr = {lr: .3e}')

# Main function to set up and run training
def main(config):
    training_config = config["training"]
    lora_config = config["lora"]

    set_seed(1234)  # Set random seed for reproducibility
    if not os.path.exists(training_config["save_path"]):
        os.makedirs(training_config["save_path"])
    
    accelerator = Accelerator(gradient_accumulation_steps=training_config["grad_accumulation_steps"])
    logger.info(f"Using device: {accelerator.device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(training_config["model_name"], 
                                                 quantization_config=quantization_config, 
                                                 torch_dtype=torch.float16)
    accelerator.print(model)

    train_dataloader = DataLoader(
        data['train'], shuffle=True, collate_fn=collate_fn, batch_size=training_config["batch_size"]
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=lora_config["r"], lora_alpha=lora_config["lora_alpha"], 
        lora_dropout=lora_config["lora_dropout"], target_modules=lora_config["target_modules"]
    )
    model = get_peft_model(model, peft_config)

    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    lr = training_config["learning_rate"] * accelerator.num_processes * accelerator.gradient_accumulation_steps
    optimizer = AdamW(params=model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * training_config["num_epochs"] * accelerator.num_processes * accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_config["num_warmup_steps"],
        num_training_steps=total_steps * 1.1,
    )

    current_epochs = 0
    global overall_step
    overall_step = 0

    # Load checkpoint if exists
    if os.path.isfile(training_config["ckpt_path"]):
        model, optimizer, lr_scheduler, current_epochs = \
            load_checkpoint(training_config["ckpt_path"], model, optimizer, lr_scheduler)
        overall_step = int(re.search('(\d)+', training_config["ckpt_path"]).group())
        logger.info(
            f'Checkpoint {training_config["ckpt_path"]} loaded at epoch {current_epochs}, the training will resume from epoch {current_epochs + 1}!')
        current_epochs += 1

    if current_epochs >= training_config["num_epochs"]:
        raise ValueError('The num_epochs should be larger than the saved epochs!!')

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    logger.info('*********************** Start training! **************************')
    for epoch in range(current_epochs, training_config["num_epochs"]):
        train_epoch(epoch, model, accelerator, train_dataloader, 
                    training_config["checkpointing_steps"], optimizer, lr_scheduler, training_config["save_path"])

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        output_dir = f"step_{overall_step}.pth"
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint(training_config["save_path"], unwrapped_model, optimizer, lr_scheduler, 
                       epoch, overall_step)

# Run the main function
main(config)
