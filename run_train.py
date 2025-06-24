import sys
import subprocess

if __name__ == "__main__":
    args = [arg.lstrip('--') for arg in sys.argv[1:]]
    subprocess.run([
        r"D:\PG_PROJECTS\Real-Time-Fraud-Detection-Pipeline\.venv\Scripts\python.exe",
        "src/train.py"
    ] + args, check=True)
