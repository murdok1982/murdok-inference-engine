import os
import sys
import subprocess
import ssl
import urllib.request

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100 * 1024 * 1024:
        print(f"Model already exists at: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/(1024*1024):.1f} MB)")
        return 0

    print(f"Fetching benchmark model from: {MODEL_URL}")
    print(f"Target destination: {MODEL_PATH}")

    # Try curl first
    try:
        cmd = ["curl.exe", "-L", "-o", MODEL_PATH, MODEL_URL]
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100 * 1024 * 1024:
            print("\nDownload complete successfully via curl.")
            return 0
    except Exception as e:
        print(f"curl attempt failed: {e}")

    # Fallback to python with unverified context
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ctx) as response, open(MODEL_PATH, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    sys.stdout.write(f"\rDownloading: {percent}% [{downloaded/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB]")
                    sys.stdout.flush()
        print("\nDownload complete successfully.")
        return 0
    except Exception as e:
        print(f"\nFailed to download model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
