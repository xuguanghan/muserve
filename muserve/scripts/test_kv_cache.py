"""Task 0.3 验证脚本：Paged KV Cache 分配/释放测试。

单进程运行（不需要 torchrun）：
    python muserve/scripts/test_kv_cache.py
"""

import os
import torch
import torch_musa

# 单卡测试，模拟 rank=0
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "8")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import torch.distributed as dist
dist.init_process_group(backend="mccl", rank=0, world_size=1)

from muserve.kv_cache import PagedKVCache, SequenceKVState
from muserve.config import KV_PAGE_SIZE, NUM_LAYERS


def test_alloc_free():
    print("[test_kv_cache] Test 1: alloc/free cycle")
    cache = PagedKVCache(num_gpu_pages=2000)

    # 分配 1000 页
    pages_a = cache.alloc_pages(1000)
    assert len(pages_a) == 1000
    assert cache.num_free_pages == 1000
    print(f"  ✓ alloc 1000 pages, free={cache.num_free_pages}")

    # 释放 500 页
    cache.free_pages(pages_a[:500])
    assert cache.num_free_pages == 1500
    print(f"  ✓ free 500 pages, free={cache.num_free_pages}")

    # 再分配 600 页（需要 600，有 1500 空闲）
    pages_b = cache.alloc_pages(600)
    assert len(pages_b) == 600
    assert cache.num_free_pages == 900
    print(f"  ✓ alloc 600 pages, free={cache.num_free_pages}")

    # OOM 测试
    try:
        cache.alloc_pages(10000)
        print("  ✗ OOM not raised!")
        return False
    except RuntimeError as e:
        print(f"  ✓ OOM correctly raised: {e}")

    print("  PASSED")
    return True


def test_kv_write_read():
    print("[test_kv_cache] Test 2: KV write/read correctness")
    cache = PagedKVCache(num_gpu_pages=100)
    device = cache.device

    kv_heads = cache.cache.shape[4]
    layer_idx = 0
    page_idx = cache.alloc_pages(1)[0]

    # 写入 KV
    k = torch.randn(kv_heads, cache.cache.shape[5], device=device, dtype=cache.cache.dtype)
    v = torch.randn(kv_heads, cache.cache.shape[5], device=device, dtype=cache.cache.dtype)
    cache.write_kv(layer_idx, page_idx, slot=0, k=k, v=v)
    torch.musa.synchronize()

    # 读回验证
    k_cache, v_cache = cache.get_kv_by_layer(layer_idx)
    k_read = k_cache[page_idx, 0]
    v_read = v_cache[page_idx, 0]

    k_err = (k_read - k).abs().max().item()
    v_err = (v_read - v).abs().max().item()
    assert k_err == 0.0, f"K mismatch: {k_err}"
    assert v_err == 0.0, f"V mismatch: {v_err}"
    print(f"  ✓ KV write/read exact match (k_err={k_err}, v_err={v_err})")
    print("  PASSED")
    return True


def test_sequence_state():
    print("[test_kv_cache] Test 3: SequenceKVState page management")
    cache = PagedKVCache(num_gpu_pages=100)
    seq = SequenceKVState()

    # 模拟 prefill 33 个 token（需要 3 页，page_size=16）
    for i in range(33):
        if seq.needs_new_page():
            page = cache.alloc_pages(1)[0]
            seq.page_table.append(page)
        seq.seq_len += 1

    assert len(seq.page_table) == 3, f"Expected 3 pages, got {len(seq.page_table)}"
    assert seq.seq_len == 33
    assert seq.last_page_slot == 0  # 33 % 16 = 1, slot index = 0 (0-indexed)
    print(f"  ✓ 33 tokens → {len(seq.page_table)} pages, last_slot={seq.last_page_slot}")

    # page_table tensor
    pt = seq.to_page_table_tensor(max_pages=16, device=cache.device)
    assert pt.shape == (1, 16)
    assert pt[0, 0].item() == seq.page_table[0]
    assert pt[0, 3].item() == 0  # 未使用的位置
    print(f"  ✓ page_table tensor shape={tuple(pt.shape)}")
    print("  PASSED")
    return True


def main():
    results = []
    results.append(test_alloc_free())
    results.append(test_kv_write_read())
    results.append(test_sequence_state())

    all_ok = all(results)
    status = "PASSED" if all_ok else "FAILED"
    print(f"\n[Task 0.3] KV Cache verification: {status}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
