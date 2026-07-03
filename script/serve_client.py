"""Unified load-test client for BOTH serving backends — same traffic, comparable numbers.

    # against Ray Serve (default http://127.0.0.1:8000/transcribe)
    python script/serve_client.py --backend ray --wav-dir /path/to/wavs -c 32 -n 512

    # against Triton  (default http://127.0.0.1:8000, model granite_asr)
    python script/serve_client.py --backend triton --wav-dir /path/to/wavs -c 32 -n 512

    # no wavs handy? synthesize N clips of 2-15 s of band-limited noise
    python script/serve_client.py --backend ray --synth -c 32 -n 512

Reports: served-audio RTFx (total audio seconds / wall seconds), latency p50/p95/p99,
error count. Run the SAME -c/-n against both backends for an apples-to-apples read.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import random
import sys
import time


def _load_wavs(args) -> list[bytes]:
    import numpy as np
    import soundfile as sf
    blobs: list[bytes] = []
    if args.synth or not args.wav_dir:
        rng = np.random.default_rng(0)
        for _ in range(min(args.n, 64)):  # cycle through 64 distinct synthetic clips
            dur = rng.uniform(2.0, 15.0)
            wav = (rng.standard_normal(int(16000 * dur)) * 0.05).astype("float32")
            buf = io.BytesIO()
            sf.write(buf, wav, 16000, format="WAV")
            blobs.append(buf.getvalue())
    else:
        for name in sorted(os.listdir(args.wav_dir)):
            if name.lower().endswith((".wav", ".flac", ".ogg")):
                with open(os.path.join(args.wav_dir, name), "rb") as f:
                    blobs.append(f.read())
        if not blobs:
            sys.exit(f"no audio files in {args.wav_dir}")
    return blobs


def _audio_seconds(blob: bytes) -> float:
    import soundfile as sf
    info = sf.info(io.BytesIO(blob))
    return info.frames / info.samplerate


async def _run_ray(args, blobs):
    import httpx
    url = args.url or "http://127.0.0.1:8000/transcribe"
    sem = asyncio.Semaphore(args.concurrency)
    lat: list[float] = []
    errs = 0
    err_samples: list[str] = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        async def one(i: int):
            nonlocal errs
            blob = blobs[i % len(blobs)]
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, content=blob)
                    r.raise_for_status()
                    lat.append(time.perf_counter() - t0)
                except Exception as e:
                    errs += 1
                    if len(err_samples) < 3:
                        body = getattr(getattr(e, "response", None), "text", "")
                        err_samples.append(f"{type(e).__name__}: {e} {body[:200]}")

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.n)))
        wall = time.perf_counter() - t0
    for msg in err_samples:
        print(f"  ERR-SAMPLE: {msg}")
    return wall, lat, errs


def _prep_pcms(blobs):
    import numpy as np
    import soundfile as sf
    pcms = []
    for blob in blobs:
        arr, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        pcms.append(np.ascontiguousarray(arr, dtype=np.float32))
    return pcms


def _run_triton_http(args, blobs):
    """Single gevent-based client + async_infer sliding window (tritonclient.http is built on
    gevent and must NOT be shared across OS threads — that deadlocks)."""
    import tritonclient.http as tc

    url = (args.url or "127.0.0.1:8000").replace("http://", "")
    lat: list[float] = []
    errs = 0
    err_samples: list[str] = []
    pcms = _prep_pcms(blobs)

    client = tc.InferenceServerClient(url=url, concurrency=args.concurrency)
    inflight: list[tuple[object, float]] = []

    def drain(block_until_below: int):
        nonlocal errs
        while len(inflight) >= max(1, block_until_below):
            handle, t0 = inflight.pop(0)
            try:
                handle.get_result()
                lat.append(time.perf_counter() - t0)
            except Exception as e:
                errs += 1
                if len(err_samples) < 3:
                    err_samples.append(f"{type(e).__name__}: {e}")

    t0_all = time.perf_counter()
    for i in range(args.n):
        pcm = pcms[i % len(pcms)]
        # max_batch_size>0: first dim is the batch dim -> shape [1, T]
        inp = tc.InferInput("AUDIO", [1, len(pcm)], "FP32")
        inp.set_data_from_numpy(pcm.reshape(1, -1))
        inflight.append((client.async_infer("granite_asr", inputs=[inp]), time.perf_counter()))
        drain(args.concurrency)
    drain(1)
    wall = time.perf_counter() - t0_all
    client.close()
    for msg in err_samples:
        print(f"  ERR-SAMPLE: {msg}")
    return wall, lat, errs


def _run_triton_grpc(args, blobs):
    """gRPC path (port 8001): callback-based async_infer with a semaphore window.
    --shm additionally moves input tensors through system shared memory (client and
    server must be on the same host) — one pre-registered region per in-flight slot."""
    import queue
    import threading

    import numpy as np
    import tritonclient.grpc as tg

    url = (args.url or "127.0.0.1:8001").replace("http://", "")
    lat: list[float] = []
    errs = 0
    err_samples: list[str] = []
    pcms = _prep_pcms(blobs)
    client = tg.InferenceServerClient(url=url)

    slots: "queue.Queue[int]" = queue.Queue()
    regions = []
    if args.shm:
        import tritonclient.utils.shared_memory as shm
        region_bytes = max(p_.nbytes for p_ in pcms)
        client.unregister_system_shared_memory()
        for k in range(args.concurrency):
            name, key = f"audio_shm_{k}", f"/audio_shm_{k}"
            handle = shm.create_shared_memory_region(name, key, region_bytes)
            client.register_system_shared_memory(name, key, region_bytes)
            regions.append(handle)
            slots.put(k)

    win = threading.Semaphore(args.concurrency)
    done = threading.Event()
    lock = threading.Lock()
    remaining = [args.n]

    def make_cb(t0, slot):
        def cb(result, error):
            nonlocal errs
            with lock:
                if error is None:
                    lat.append(time.perf_counter() - t0)
                else:
                    errs += 1
                    if len(err_samples) < 3:
                        err_samples.append(f"{type(error).__name__}: {error}")
                remaining[0] -= 1
                if remaining[0] == 0:
                    done.set()
            if slot is not None:
                slots.put(slot)
            win.release()
        return cb

    t0_all = time.perf_counter()
    for i in range(args.n):
        pcm = pcms[i % len(pcms)]
        win.acquire()
        inp = tg.InferInput("AUDIO", [1, len(pcm)], "FP32")
        slot = None
        if args.shm:
            import tritonclient.utils.shared_memory as shm
            slot = slots.get()
            shm.set_shared_memory_region(regions[slot], [pcm.reshape(1, -1)])
            inp.set_shared_memory(f"audio_shm_{slot}", pcm.nbytes)
        else:
            inp.set_data_from_numpy(pcm.reshape(1, -1))
        client.async_infer("granite_asr", inputs=[inp],
                           callback=make_cb(time.perf_counter(), slot))
    done.wait(timeout=600)
    wall = time.perf_counter() - t0_all

    if args.shm:
        import tritonclient.utils.shared_memory as shm
        client.unregister_system_shared_memory()
        for h in regions:
            shm.destroy_shared_memory_region(h)
    client.close()
    for msg in err_samples:
        print(f"  ERR-SAMPLE: {msg}")
    return wall, lat, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ray", "triton"], required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("-c", "--concurrency", type=int, default=16)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--protocol", choices=["http", "grpc"], default="http",
                    help="triton transport (grpc uses port 8001 by default)")
    ap.add_argument("--shm", action="store_true",
                    help="triton+grpc only: system shared-memory input transport (same-host)")
    args = ap.parse_args()

    blobs = _load_wavs(args)
    random.seed(0)
    audio_s = sum(_audio_seconds(blobs[i % len(blobs)]) for i in range(args.n))

    if args.backend == "ray":
        wall, lat, errs = asyncio.run(_run_ray(args, blobs))
    elif args.protocol == "grpc" or args.shm:
        wall, lat, errs = _run_triton_grpc(args, blobs)
    else:
        wall, lat, errs = _run_triton_http(args, blobs)

    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] * 1e3 if lat else float("nan")
    tag = args.backend if args.backend == "ray" else \
        f"{args.backend}-{'grpc' if (args.protocol == 'grpc' or args.shm) else 'http'}{'+shm' if args.shm else ''}"
    print(f"[{tag}] n={args.n} c={args.concurrency} errors={errs}")
    print(f"  wall={wall:.1f}s  audio={audio_s:.0f}s  served-RTFx={audio_s / wall:.1f}")
    print(f"  latency ms: p50={pct(.50):.0f}  p95={pct(.95):.0f}  p99={pct(.99):.0f}")


if __name__ == "__main__":
    main()
