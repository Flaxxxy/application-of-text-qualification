"""Measure the website's peak memory (RAM) the way Render will run it: gunicorn, 1 worker.

Usage:  python scripts/check_memory.py

Starts gunicorn on a free port, then does what a visitor does: opens every page, checks one order
with a photo, runs the 12 demo orders as a batch. Meanwhile it samples the memory (RSS) of the
gunicorn processes every 0.1 s. Render's free tier has 512 MB; our target is < 450 MB.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
TARGET_MB = 450


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tree_rss_mb(proc: psutil.Process) -> float:
    """Memory of gunicorn's master process plus its worker(s)."""
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 1e6


def post_multipart(url: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> int:
    boundary = "----check-memory"
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data, ctype) in files.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def main() -> None:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    gunicorn = subprocess.Popen([sys.executable, "-m", "gunicorn", "app.app:app", "--workers", "1",
                                 "--timeout", "120", "--bind", f"127.0.0.1:{port}"], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = psutil.Process(gunicorn.pid)
    samples: list[tuple[float, str, float]] = []
    step = {"name": "starting"}
    stop = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            samples.append((time.time() - t0, step["name"], tree_rss_mb(proc)))
            time.sleep(0.1)

    threading.Thread(target=sampler, daemon=True).start()
    try:
        t0 = time.time()
        while True:  # wait until the models are loaded and /health answers
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except OSError:
                if time.time() - t0 > 120:
                    raise SystemExit("gunicorn did not start within 120 s")
                time.sleep(0.5)
        startup_s = time.time() - t0
        after_start = tree_rss_mb(proc)

        for name, path in [("home", "/"), ("check page", "/check"), ("models", "/models"), ("history", "/history")]:
            step["name"] = name
            urllib.request.urlopen(base + path, timeout=60).read()
        step["name"] = "check order with photo"
        photo = (ROOT / "demo_data" / "images" / "DEMO-01.jpg").read_bytes()
        t = time.time()
        post_multipart(base + "/check", {"review_text": "The box arrived crushed and I want a refund.",
                                         "category": "electronics"}, {"photo": ("p.jpg", photo, "image/jpeg")})
        one_order_s = time.time() - t
        step["name"] = "batch: 12 demo orders"
        t = time.time()
        post_multipart(base + "/batch", {"action": "demo"}, {})
        batch_s = time.time() - t
        step["name"] = "idle"
        time.sleep(1)
    finally:
        stop.set()
        gunicorn.terminate()
        gunicorn.wait(timeout=30)

    peak = max(samples, key=lambda s: s[2])
    print(f"start-up (models loaded):  {startup_s:5.1f} s, {after_start:6.1f} MB")
    print(f"one order with photo:      {one_order_s:5.2f} s")
    print(f"batch of 12 demo orders:   {batch_s:5.2f} s")
    print(f"PEAK memory:               {peak[2]:6.1f} MB (during: {peak[1]})")
    print(f"target < {TARGET_MB} MB -> {'OK' if peak[2] < TARGET_MB else 'TOO HIGH'}")


if __name__ == "__main__":
    main()
