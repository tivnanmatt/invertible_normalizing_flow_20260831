"""Resumable HTTP fetch with retries (Range requests). Usage: fetch_resumable.py URL DEST"""
import os
import sys
import time
import urllib.request

url, dest = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(dest), exist_ok=True)
tmp = dest + ".part"
for attempt in range(30):
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as r:
            total = have + int(r.headers.get("Content-Length", 0))
            mode = "ab" if have and r.status == 206 else "wb"
            with open(tmp, mode) as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        os.rename(tmp, dest)
        print(f"OK {dest} {os.path.getsize(dest)} bytes")
        sys.exit(0)
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {e} (have {have} bytes)", flush=True)
        time.sleep(3)
print("FAILED", dest)
sys.exit(1)
