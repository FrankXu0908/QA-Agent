
from pathlib import Path
import sys
# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = Path(__file__).resolve().parent.parent
import subprocess

MODEL_PATH = parent_dir/"models/Qwen2.5-32B-Instruct-AWQ"
PORT = 8001
# 构造命令行
cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--port", str(PORT),
    "--tensor_parallel_size", "2",  # 并行使用 2 张 GPU
    "--gpu_memory_utilization", "0.8",  # 每张 GPU 使用 80% 显存
]
# 启动子进程
subprocess.run(cmd)