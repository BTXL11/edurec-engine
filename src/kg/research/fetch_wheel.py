"""Download a large wheel with parallel HTTP range requests, then verify it.

Single-connection downloads from download.pytorch.org are throttled here
(~0.1-3.5 MB/s, both endpoints), which turns a 2.8 GB CUDA wheel into a
30-60 minute stall. Eight concurrent range requests each get their own
connection, so total throughput is much higher.

Usage:
    python fetch_wheel.py <url> <destination.whl> [threads]
"""
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "pip/26"}


def head_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"]), resp.headers.get("Accept-Ranges", "")


def download_part(url, start, end, path, results, index, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            got = os.path.getsize(path)
            want = end - start + 1
            if got != want:
                raise IOError(f"short read {got} != {want}")
            results[index] = got
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                results[index] = f"FAIL {type(exc).__name__}: {exc}"
                return
            time.sleep(2 * attempt)


def main():
    url, dest = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(dest)), "_parts")
    os.makedirs(tmpdir, exist_ok=True)

    size, accept = head_size(url)
    print(f"remote size: {size/1048576:.1f} MB (accept-ranges={accept!r})", flush=True)
    if size <= 0:
        print("no content-length; aborting")
        return 1

    per = size // threads
    bounds = []
    for i in range(threads):
        start = i * per
        end = size - 1 if i == threads - 1 else (i + 1) * per - 1
        bounds.append((start, end))

    results = [None] * threads
    workers = []
    t0 = time.time()
    for i, (start, end) in enumerate(bounds):
        part = os.path.join(tmpdir, f"part{i:02d}")
        th = threading.Thread(target=download_part, args=(url, start, end, part, results, i))
        th.start()
        workers.append(th)

    while any(th.is_alive() for th in workers):
        done = sum(os.path.getsize(os.path.join(tmpdir, f"part{i:02d}"))
                   for i in range(threads) if os.path.exists(os.path.join(tmpdir, f"part{i:02d}")))
        rate = done / 1048576 / max(time.time() - t0, 1e-3)
        print(f"  {done/1048576:8.1f}/{size/1048576:.1f} MB  {rate:6.2f} MB/s", flush=True)
        time.sleep(10)

    for th in workers:
        th.join()

    bad = [r for r in results if not isinstance(r, int)]
    if bad:
        print("failures:", bad)
        return 1

    print(f"all parts done in {time.time()-t0:.0f}s; concatenating", flush=True)
    with open(dest, "wb") as out:
        for i in range(threads):
            part = os.path.join(tmpdir, f"part{i:02d}")
            with open(part, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
            os.remove(part)
    os.rmdir(tmpdir)

    written = os.path.getsize(dest)
    print(f"written {written/1048576:.1f} MB to {dest}", flush=True)
    if written != size:
        print("SIZE MISMATCH")
        return 1

    # pip prints the sha256 fragment in the index page; verify it if given via env
    expect = os.environ.get("EXPECT_SHA256")
    if expect:
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        got = h.hexdigest()
        print(f"sha256 {got}")
        if got != expect:
            print("HASH MISMATCH")
            return 1
        print("sha256 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
