#!/usr/bin/env python3
"""evalscope 10-round benchmark with high KV cache hit rate.

Pre-generates ALL prompts before timing starts.
Uses custom dataset with pre-built prompt file to avoid tokenizer.decode() overhead during benchmark.
Prefix: ~25K tokens (shared, cached after first round)
Suffix: ~3K tokens (unique per request)
Total: ~28K tokens, ~89% hit rate.
"""

import json, os, sys, time

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "/data/models/qwen3.5fp8"
TOKENIZER = "/data/models/qwen3.5fp8"
TOTAL_TOKENS = 28000
SHARED_RATIO = 0.89
SHARED_TOKENS = int(TOTAL_TOKENS * SHARED_RATIO)
UNIQUE_TOKENS = TOTAL_TOKENS - SHARED_TOKENS
OUTPUT = 128
ROUNDS = 10
CONC = 40
TEMP = 0.0
OUTPUT_DIR = "/data/loadpup_output"
WARMUP_ROUNDS = 3

# ========== Phase 1: Pre-generate all prompts (not timed) ==========
print("Phase 1: Pre-generating prompts ...", flush=True)
t_pre = time.perf_counter()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

base_text = ("The field of artificial intelligence has undergone remarkable "
             "transformations over the past decade. Deep learning models, "
             "particularly transformers, have revolutionized natural language "
             "processing, computer vision, and multimodal applications. ") * 2000

all_tokens = tokenizer.encode(base_text)
while len(all_tokens) < SHARED_TOKENS:
    all_tokens = all_tokens + all_tokens
shared_tokens = all_tokens[:SHARED_TOKENS]
shared_text = tokenizer.decode(shared_tokens)

prompts = []
max_char_len = 0
for i in range(CONC):
    unique_text = f"Request {i}: " + base_text[:UNIQUE_TOKENS * 4]
    combined = shared_text + "\n\n" + unique_text + "\n\nSummarize key AI developments."
    toks = tokenizer.encode(combined)
    if len(toks) > TOTAL_TOKENS:
        toks = toks[:TOTAL_TOKENS]
    prompt_text = tokenizer.decode(toks).replace("\n", " ")
    prompts.append(prompt_text)
    max_char_len = max(max_char_len, len(prompt_text))

sample_tokens = len(tokenizer.encode(prompts[0]))
pre_time = time.perf_counter() - t_pre
print(f"  Generated {CONC} prompts: {sample_tokens} tokens, {max_char_len} chars each, took {pre_time:.1f}s", flush=True)

# Save to plain text file (one prompt per line)
prompts_file = f"{OUTPUT_DIR}/prefix_cached_prompts.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(prompts_file, "w") as f:
    for p in prompts:
        f.write(p + "\n")

# ========== Phase 2: Benchmark (timed) ==========
print(f"\nConfig: {TOTAL_TOKENS} total, {SHARED_TOKENS} shared ({SHARED_RATIO*100:.0f}%), "
      f"{UNIQUE_TOKENS} unique, {CONC}c, {ROUNDS} rounds", flush=True)

from evalscope.perf.main import run_perf_benchmark


def run_round(round_idx):
    task_cfg = {
        "url": URL,
        "parallel": CONC,
        "model": MODEL,
        "number": CONC,
        "api": "openai",
        "tokenizer_path": TOKENIZER,
        "max_prompt_length": max_char_len + 1000,
        "min_prompt_length": 0,
        "max_tokens": OUTPUT,
        "min_tokens": OUTPUT,
        "temperature": TEMP,
        "stream": True,
        "seed": 42,
        "dataset": "custom",
        "dataset_path": prompts_file,
        "apply_chat_template": True,
        "no_test_connection": True,
        "outputs_dir": f"{OUTPUT_DIR}/cached_40c/round_{round_idx}",
    }
    t0 = time.perf_counter()
    result = run_perf_benchmark(task_cfg)
    elapsed = time.perf_counter() - t0

    summary, percentiles = {}, {}
    if isinstance(result, tuple) and len(result) >= 2:
        summary = result[0] if isinstance(result[0], dict) else {}
        percentiles = result[1] if isinstance(result[1], dict) else {}

    ttft_p50 = ttft_p90 = ttft_p99 = 0
    lat_p50 = lat_p99 = 0
    p_keys = percentiles.get("Percentiles", [])
    ttft_vals = percentiles.get("TTFT (s)", [])
    lat_vals = percentiles.get("Latency (s)", [])
    for i, p in enumerate(p_keys):
        ps = str(p)
        if ps == "50%" and i < len(ttft_vals):
            ttft_p50 = ttft_vals[i] * 1000
            lat_p50 = lat_vals[i] * 1000 if i < len(lat_vals) else 0
        elif ps == "90%" and i < len(ttft_vals):
            ttft_p90 = ttft_vals[i] * 1000
        elif ps == "99%" and i < len(ttft_vals):
            ttft_p99 = ttft_vals[i] * 1000
            lat_p99 = lat_vals[i] * 1000 if i < len(lat_vals) else 0

    return {
        "round": round_idx,
        "wall_time_s": round(elapsed, 2),
        "throughput_tps": round(summary.get("Output token throughput (tok/s)", 0), 1),
        "total_throughput_tps": round(summary.get("Total token throughput (tok/s)", 0), 1),
        "ttft_avg_ms": round(summary.get("Average time to first token (s)", 0) * 1000, 1),
        "ttft_p50_ms": round(ttft_p50, 1),
        "ttft_p90_ms": round(ttft_p90, 1),
        "ttft_p99_ms": round(ttft_p99, 1),
        "tpot_avg_ms": round(summary.get("Average time per output token (s)", 0) * 1000, 1),
        "latency_avg_ms": round(summary.get("Average latency (s)", 0) * 1000, 1),
        "latency_p50_ms": round(lat_p50, 1),
        "latency_p99_ms": round(lat_p99, 1),
        "avg_input_tokens": round(summary.get("Average input tokens per request", 0), 0),
        "avg_output_tokens": round(summary.get("Average output tokens per request", 0), 0),
        "success": summary.get("Succeed requests", 0),
        "total": summary.get("Total requests", 0),
        "failed": summary.get("Failed requests", 0),
    }


# Warmup
print(f"Warmup ({WARMUP_ROUNDS} rounds) ...", flush=True)
for wi in range(WARMUP_ROUNDS):
    try:
        run_perf_benchmark({
            "url": URL, "parallel": 4, "model": MODEL, "number": 4,
            "api": "openai", "tokenizer_path": TOKENIZER,
            "max_prompt_length": 1024, "min_prompt_length": 1024,
            "max_tokens": 64, "min_tokens": 64, "temperature": 0,
            "stream": True, "dataset": "random", "apply_chat_template": True,
            "no_test_connection": True, "outputs_dir": f"{OUTPUT_DIR}/cached_warmup_{wi}",
        })
    except Exception:
        pass
    print(f"  {wi+1}/{WARMUP_ROUNDS}", flush=True)
time.sleep(2)

# Main 10 rounds
print(f"\n{'='*70}", flush=True)
print(f"  {CONC}c | {TOTAL_TOKENS//1000}K tokens | {SHARED_TOKENS//1000}K shared ({SHARED_RATIO*100:.0f}%) | {OUTPUT} output | temp={TEMP}", flush=True)
print(f"{'='*70}", flush=True)
rounds = []
for rd in range(1, ROUNDS + 1):
    print(f"  Round {rd}/{ROUNDS}...", end="", flush=True)
    try:
        m = run_round(rd)
        rounds.append(m)
        print(f" TPS={m['throughput_tps']:.1f} TTFT_P50={m['ttft_p50_ms']:.0f}ms "
              f"TTFT_P99={m['ttft_p99_ms']:.0f}ms wall={m['wall_time_s']:.1f}s "
              f"ok={m['success']}/{m['total']}", flush=True)
    except Exception as e:
        print(f" ERROR: {e}", flush=True)

# Summary table
print(f"\n{'='*80}")
print(f"evalscope Prefix-Cached Benchmark | {TOTAL_TOKENS//1000}K tokens | "
      f"{SHARED_TOKENS//1000}K shared ({SHARED_RATIO*100:.0f}%) | {OUTPUT} output | temp={TEMP}")
print(f"{'='*80}")
print(f"{'Rd':>3} | {'TPS':>7} | {'TTFT_P50':>9} | {'TTFT_P90':>9} | {'TTFT_P99':>9} | {'TPOT_avg':>9} | {'Wall':>6} | {'OK':>5}")
print(f"{'-'*3}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}-+-{'-'*5}")
for r in rounds:
    print(f"{r['round']:>3} | {r['throughput_tps']:>7.1f} | {r['ttft_p50_ms']:>8.1f}ms | "
          f"{r['ttft_p90_ms']:>8.1f}ms | {r['ttft_p99_ms']:>8.1f}ms | "
          f"{r['tpot_avg_ms']:>8.1f}ms | {r['wall_time_s']:>6.1f} | {r['success']:>5}")

tps = [r["throughput_tps"] for r in rounds]
p50 = [r["ttft_p50_ms"] for r in rounds]
p99 = [r["ttft_p99_ms"] for r in rounds]
if tps:
    avg_tps = sum(tps) / len(tps)
    avg_p50 = sum(p50) / len(p50)
    avg_p99 = sum(p99) / len(p99)
    print(f"{'-'*3}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}-+-{'-'*5}")
    print(f"{'AVG':>3} | {avg_tps:>7.1f} | {avg_p50:>8.1f}ms | {'':>9} | {avg_p99:>8.1f}ms |")

# Save
outfile = f"{OUTPUT_DIR}/evalscope_cached_40c_28k.json"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    json.dump({
        "config": {"concurrency": CONC, "context": TOTAL_TOKENS, "shared": SHARED_TOKENS,
                   "hit_rate": SHARED_RATIO, "output": OUTPUT,
                   "rounds": ROUNDS, "temperature": TEMP, "tool": "evalscope", "warmup": WARMUP_ROUNDS},
        "results": rounds
    }, f, indent=2)
print(f"\nSaved to {outfile}")
