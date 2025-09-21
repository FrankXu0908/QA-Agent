from pathlib import Path
import subprocess
from pathlib import Path
raw_path = Path("现场流程相关文件.zip")
staging_path = Path("raw/")
staging_path.mkdir(parents=True, exist_ok=True)
try:
    # Linux/macOS
    subprocess.run(["unzip", "-O", "GBK", raw_path, "-d", str(staging_path)], check=True)
except FileNotFoundError:
    print("解压失败")
