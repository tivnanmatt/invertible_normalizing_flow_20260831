"""Parallel ranged HTTP fetch. Usage: fetch_parallel.py URL DEST [N_CONN]"""
import os
import sys
import threading
import urllib.request
UA = sys.argv[4] if len(sys.argv) > 4 else "Mozilla/5.0"

url, dest = sys.argv[1], sys.argv[2]
n_conn = int(sys.argv[3]) if len(sys.argv) > 3 else 16
os.makedirs(os.path.dirname(dest), exist_ok=True)

req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers["Content-Length"])
print(f"{total} bytes, {n_conn} connections", flush=True)

chunk = (total + n_conn - 1) // n_conn
lock = threading.Lock()
done = [0]
fails = []

with open(dest + ".part", "wb") as f:
    f.truncate(total)

def worker(i):
    lo, hi = i * chunk, min((i + 1) * chunk, total) - 1
    for attempt in range(20):
        try:
            got = 0
            rq = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(rq, timeout=60) as r, open(dest + ".part", "r+b") as f:
                f.seek(lo)
                while got <= hi - lo:
                    c = r.read(1 << 18)
                    if not c:
                        break
                    f.write(c)
                    got += len(c)
            if got >= hi - lo + 1:
                with lock:
                    done[0] += 1
                    print(f"chunk {i} done ({done[0]}/{n_conn})", flush=True)
                return
            lo += got  # partial: resume from where this chunk stalled
        except Exception as e:
            print(f"chunk {i} attempt {attempt}: {type(e).__name__}", flush=True)
    fails.append(i)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_conn)]
[t.start() for t in threads]
[t.join() for t in threads]
if fails:
    print("FAILED chunks:", fails)
    sys.exit(1)
os.rename(dest + ".part", dest)
print(f"OK {dest} {os.path.getsize(dest)} bytes")
