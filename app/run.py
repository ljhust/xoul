import ray
from ray import serve
from deployment.llm import build_app_from_yaml  # 替换为您的代码文件名和路径

# 启动 Ray 和 Serve
ray.init()  # 如果连接到远程集群，可以传入 address 参数，例如 ray.init(address="auto")
serve.start()

# 加载 YAML 配置并启动服务
config_path = "llm.yaml"  # 替换为您的 YAML 文件路径
serve_app = build_app_from_yaml(config_path)

# 部署 Serve 应用
serve.run(serve_app)

print("Serve application is running. You can make requests to the endpoints.")
