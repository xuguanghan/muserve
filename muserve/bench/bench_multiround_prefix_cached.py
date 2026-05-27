#!/usr/bin/env python3
"""Multi-round benchmark with high KV cache hit rate.

Uses a shared long prefix + unique short suffix per request.
Prefix: ~25K tokens (shared across requests, cached after first round)
Suffix: ~3K tokens (unique per request, must be prefilled each time)
Total: ~28K tokens per request, ~89% KV cache hit rate after first round.
"""

import time, sys, os, json, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
ROUNDS = int(os.environ.get("BENCH_ROUNDS", 10))
N = int(os.environ.get("BENCH_N", 40))
TOTAL_TOKENS = int(os.environ.get("BENCH_CTX", 28000))
SHARED_RATIO = float(os.environ.get("BENCH_SHARED_RATIO", 0.89))
OUTPUT = int(os.environ.get("BENCH_OUT", 128))
WARMUP = int(os.environ.get("BENCH_WARMUP", 3))
MODEL_PATH = os.environ.get("BENCH_MODEL", "/data/models/qwen3.5fp8")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Build shared prefix (~25K tokens)
shared_tokens_count = int(TOTAL_TOKENS * SHARED_RATIO)
unique_tokens_count = TOTAL_TOKENS - shared_tokens_count

base_text = ("The field of artificial intelligence has undergone remarkable "
             "transformations over the past decade. Deep learning models, "
             "particularly transformers, have revolutionized natural language "
             "processing, computer vision, and multimodal applications. ") * 2000

# Shared prefix: token-level precise
all_tokens = tokenizer.encode(base_text)
while len(all_tokens) < shared_tokens_count:
    all_tokens = all_tokens + all_tokens
shared_tokens = all_tokens[:shared_tokens_count]
shared_text = tokenizer.decode(shared_tokens)

print(f"Config: {TOTAL_TOKENS} total tokens, {shared_tokens_count} shared ({SHARED_RATIO*100:.0f}%), "
      f"{unique_tokens_count} unique, {N}c, {ROUNDS} rounds", flush=True)
print(f"Shared prefix: {shared_tokens_count} tokens", flush=True)


def build_prompt(req_id):
    """Build prompt with shared prefix + unique suffix."""
    # Generate unique content per request
    unique_text = f"Request {req_id}: " + base_text[:unique_tokens_count * 4]
    # Trim to exact token count
    combined = shared_text + "\n\n" + unique_text + "\n\nSummarize key AI developments."
    tokens = tokenizer.encode(combined)
    if len(tokens) > TOTAL_TOKENS:
        tokens = tokens[:TOTAL_TOKENS]
        combined = tokenizer.decode(tokens)
    return combined, len(tokens)


# Pre-build all prompts
prompts = []
for i in range(N):
    p, t = build_prompt(i)
    prompts.append(p)
print(f"Sample prompt tokens: {t}", flush=True)


def stream(prompt):
    t0 = time.perf_counter()
    ttft = None
    tok = 0
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": OUTPUT, "temperature": 0, "stream": True,
        }, timeout=1200, stream=True)
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                if line[6:] == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tok += 1
    except Exception as e:
        return {"ttft": None, "tokens": 0, "error": str(e)}
    return {"ttft": ttft, "tokens": tok}


# Warmup (uses shared prefix to populate KV cache)
print(f"Warmup ({WARMUP} rounds) ...", flush=True)
for i in range(WARMUP):
    stream("Hello, how are you?")
    print(f"  {i+1}/{WARMUP}", flush=True)
time.sleep(2)

all_rounds = []

for rd in range(1, ROUNDS + 1):
    print(f"\n--- Round {rd}/{ROUNDS} ---", flush=True)
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(N) as ex:
        fs = [ex.submit(stream, prompts[i % len(prompts)]) for i in range(N)]
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
        "input_tokens": TOTAL_TOKENS,
        "shared_tokens": shared_tokens_count,
        "hit_rate_pct": round(shared_tokens_count / TOTAL_TOKENS * 100, 1),
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
          f"Time: {summary['wall_time_s']}s", flush=True)

# Summary table
print(f"\n{'='*80}")
print(f"bench_multiround_prefix_cached | {TOTAL_TOKENS//1000}K tokens | "
      f"{shared_tokens_count//1000}K shared ({SHARED_RATIO*100:.0f}%) | "
      f"{N}c | {ROUNDS} rounds")
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

# Save JSON
outfile = f"/tmp/bench_multiround_cached_{N}c_{TOTAL_TOKENS//1000}k.json"
with open(outfile, "w") as f:
    json.dump({
        "config": {"api": "openai", "concurrency": N, "context_tokens": TOTAL_TOKENS,
                   "shared_tokens": shared_tokens_count, "hit_rate": SHARED_RATIO,
                   "output": OUTPUT, "rounds": ROUNDS},
        "rounds": all_rounds,
        "avg_throughput_tps": round(avg_thr, 1),
        "avg_ttft_p50_ms": round(avg_p50, 1),
        "avg_ttft_p99_ms": round(avg_p99, 1),
    }, f, indent=2)
print(f"\nResults saved to {outfile}")
