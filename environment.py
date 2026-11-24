"""Read-only environment inventory; writes only inside the project results/."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch


def capture(args):
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def environment():
    data = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                hostname=platform.node(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), cwd=os.getcwd())
    for package in ("triton", "vllm", "numpy", "matplotlib"):
        try:
            data[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data[package] = None
    data["nvidia_smi"] = capture(["nvidia-smi"])
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        data["gpu"] = dict(name=p.name, total_memory_bytes=p.total_memory,
                           capability=torch.cuda.get_device_capability(0), sms=p.multi_processor_count)
        data["memory_free_total_bytes"] = torch.cuda.mem_get_info()
    return data


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    data = environment()
    Path("results/environment.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    Path("results/pip-freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"])["stdout"])
    print(json.dumps(data, indent=2, ensure_ascii=False))
