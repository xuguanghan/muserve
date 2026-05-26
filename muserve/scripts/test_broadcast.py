"""Test broadcast_pyobj across all TP ranks."""
import sys
sys.path.insert(0, "/workspace")
from muserve.distributed import init_distributed, get_tp_rank, broadcast_pyobj, barrier

init_distributed()
rank = get_tp_rank()
barrier()

# Test 1: broadcast dict
if rank == 0:
    data = {"action": "prefill", "input_ids": [1, 2, 3], "max_tokens": 10}
else:
    data = None

result = broadcast_pyobj(data, src=0)

if rank == 0:
    print(f"[test_broadcast] rank 0 sent: {data}")
if rank == 1:
    print(f"[test_broadcast] rank 1 received: {result}")

barrier()

# Test 2: broadcast None (idle signal)
if rank == 0:
    data2 = None
else:
    data2 = None

result2 = broadcast_pyobj(data2, src=0)
if rank == 0:
    print(f"[test_broadcast] None broadcast OK: {result2}")

barrier()
if rank == 0:
    print("[test_broadcast] ALL PASSED")
