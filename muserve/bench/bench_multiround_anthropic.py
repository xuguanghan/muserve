#!/usr/bin/env python3
"""Multi-round benchmark using Anthropic API (/v1/messages) format."""

import time, sys, os, json, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
ROUNDS = int(os.environ.get("BENCH_ROUNDS", 10))

N = int(os.environ.get("BENCH_N", 60))
CTX = int(os.environ.get("BENCH_CTX", 32000))
OUTPUT = int(os.environ.get("BENCH_OUT", 128))
WARMUP = int(os.environ.get("BENCH_WARMUP", 3))
MODEL = os.environ.get("BENCH_MODEL", "qwen3.6-27b-fp8")

base_text = ("The field of artificial intelligence has undergone remarkable "
             "transformations over the past decade. Deep learning models, "
             "particularly transformers, have revolutionized natural language "
             "processing, computer vision, and multimodal applications. ") * 2000
prompt = base_text[:CTX] + "\n\nSummarize key AI developments in 3 bullet points."


def stream(p):
    t0 = time.perf_counter()
    ttft = None
    tok = 0
    try:
        r = requests.post(f"{BASE}/v1/messages", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": p}],
            "max_tokens": OUTPUT,
            "temperature": 0,
            "stream": True,
        }, timeout=1200, stream=True)
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    d = json.loads(payload)
                    if d.get("type") == "content_block_delta":
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        tok += 1
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"ttft": None, "tokens": 0, "error": str(e)}
    return {"ttft": ttft, "tokens": tok}


# Warmup
print(f"Warmup ({WARMUP} rounds) ...")
for i in range(WARMUP):
    stream("Hello, how are you?")
    print(f"  {i+1}/{WARMUP}")
time.sleep(2)

all_rounds = []

for rd in range(1, ROUNDS + 1):
    print(f"\n--- Round {rd}/{ROUNDS} ---")
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(N) as ex:
        fs = [ex.submit(stream, prompt) for _ in range(N)]
        for f in as_completed(fs):
            results.append(f.result())
    total_t = time.perf_counter() - t0

    ok = [r for r in results if r.get("ttft")]
    fail = [r for r in results if not r.get("ttft")]
    ttfts = sorted([r["ttft"] for r in ok])
    tokens = sum(r["tokens"] for r in ok)
    thr = tokens / total_t if total_t > 0 else 0

    summary = {
        "round": rd,
        "success": len(ok), "total": len(results), "failed": len(fail),
        "wall_time_s": round(total_t, 2),
        "total_output_tokens": tokens,
        "throughput_tps": round(thr, 1),
    }

    if ttfts:
        n = len(ttfts)
        summary.update({
            "ttft_min_ms": round(ttfts[0] * 1000, 1),
            "ttft_p50_ms": round(ttfts[n // 2] * 1000, 1),
            "ttft_p90_ms": round(ttfts[int(n * 0.9)] * 1000, 1),
            "ttft_p99_ms": round(ttfts[int(n * 0.99)] * 1000, 1),
            "ttft_max_ms": round(ttfts[-1] * 1000, 1),
        })

    all_rounds.append(summary)
    print(f"  Throughput: {summary['throughput_tps']} t/s | "
          f"TTFT P50: {summary.get('ttft_p50_ms', '?')}ms | "
          f"P99: {summary.get('ttft_p99_ms', '?')}ms | "
          f"Time: {summary['wall_time_s']}s")

# Summary table
print(f"\n{'='*80}")
print(f"Qwen3.6-27B Anthropic API | {CTX//1000}K context | {N} concurrent | {ROUNDS} rounds")
print(f"{'='*80}")
print(f"{'Round':>5} | {'TPS':>7} | {'P50(ms)':>8} | {'P90(ms)':>8} | {'P99(ms)':>8} | {'Min(ms)':>7} | {'Max(ms)':>7} | {'Time(s)':>7}")
print(f"{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for r in all_rounds:
    print(f"{r['round']:>5} | {r['throughput_tps']:>7} | {r.get('ttft_p50_ms', '-'):>8} | {r.get('ttft_p90_ms', '-'):>8} | {r.get('ttft_p99_ms', '-'):>8} | {r.get('ttft_min_ms', '-'):>7} | {r.get('ttft_max_ms', '-'):>7} | {r['wall_time_s']:>7}")

avg_thr = sum(r["throughput_tps"] for r in all_rounds) / len(all_rounds)
p50s = [r.get("ttft_p50_ms") for r in all_rounds if r.get("ttft_p50_ms")]
p99s = [r.get("ttft_p99_ms") for r in all_rounds if r.get("ttft_p99_ms")]
avg_p50 = sum(p50s) / len(p50s) if p50s else 0
avg_p99 = sum(p99s) / len(p99s) if p99s else 0
print(f"{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
print(f"{'AVG':>5} | {avg_thr:>7.1f} | {avg_p50:>8.1f} | {'':>8} | {avg_p99:>8.1f} | {'':>7} | {'':>7} | {'':>7}")

outfile = f"/tmp/bench_anthropic_{N}c_{CTX//1000}k.json"
with open(outfile, "w") as f:
    json.dump({"config": {"api": "anthropic", "concurrency": N, "context": CTX, "output": OUTPUT, "rounds": ROUNDS, "model": MODEL},
               "rounds": all_rounds,
               "avg_throughput_tps": round(avg_thr, 1),
               "avg_ttft_p50_ms": round(avg_p50, 1),
               "avg_ttft_p99_ms": round(avg_p99, 1)}, f, indent=2)
print(f"\nResults saved to {outfile}")
