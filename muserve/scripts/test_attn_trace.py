"""Trace attention layer internals to find the norm explosion source."""
import sys; sys.path.insert(0, "/workspace")
import torch, torch_musa
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS
from muserve.model.qwen35_layer import (
    fp8_linear, bf16_linear, rms_norm, layer_forward_prefill,
    _ATTN_Q_HEADS_LOCAL, _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM, _ATTN_Q_HEADS_LOCAL
)

def main():
    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")
    torch.musa.set_device(device)
    wi = _load_index(DEFAULT_MODEL_PATH)
    embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, wi)
    barrier()

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    ids = torch.tensor(tokenizer.encode("2+2="), device=device)
    hidden = F.embedding(ids, embed_w["embed_tokens.weight"].bfloat16())

    # Run first 3 GDN layers
    cu = torch.tensor([0, len(ids)], device=device, dtype=torch.int64)
    for i in range(3):
        w = load_layer_weights(DEFAULT_MODEL_PATH, i, wi)
        hidden, _, _ = layer_forward_prefill(hidden, cu, w)
    barrier()

    if rank == 0:
        print(f"After 3 GDN layers: h_norm={hidden.norm():.2f}", flush=True)

    # Now trace layer 3 (first attention layer) step by step
    w = load_layer_weights(DEFAULT_MODEL_PATH, 3, wi)
    norm_w = w["input_layernorm.weight"]
    x = rms_norm(hidden, norm_w)
    if rank == 0:
        print(f"\n=== Layer 3 ATTN trace ===", flush=True)
        print(f"  input_layernorm: x_norm={x.norm():.3f}", flush=True)

    # Q+gate projection
    w_q = w["attn.q_proj.weight"]; s_q = w.get("attn.q_proj.weight_scale_inv")
    qg = fp8_linear(x, w_q, s_q) if s_q is not None else bf16_linear(x, w_q)
    local_qg_dim = qg.shape[-1]
    local_q_dim = local_qg_dim // 2
    q_local = qg[..., :local_q_dim]
    gate_local = qg[..., local_q_dim:]

    # K, V projections
    w_k = w["attn.k_proj.weight"]; s_k = w.get("attn.k_proj.weight_scale_inv")
    k = fp8_linear(x, w_k, s_k) if s_k is not None else bf16_linear(x, w_k)
    w_v = w["attn.v_proj.weight"]; s_v = w.get("attn.v_proj.weight_scale_inv")
    v = fp8_linear(x, w_v, s_v) if s_v is not None else bf16_linear(x, w_v)

    if rank == 0:
        print(f"  Q proj: q_norm={q_local.norm():.2f} gate_norm={gate_local.norm():.2f}", flush=True)
        print(f"  K proj: k_norm={k.norm():.2f}", flush=True)
        print(f"  V proj: v_norm={v.norm():.2f}", flush=True)

    # Reshape to heads
    total = x.shape[0]
    q_heads = q_local.reshape(total, _ATTN_Q_HEADS_LOCAL, _ATTN_HEAD_DIM)
    k_heads = k.reshape(total, _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM)
    v_heads = v.reshape(total, _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM)

    # QK norm
    if "attn.q_norm.weight" in w:
        q_heads = rms_norm(q_heads, w["attn.q_norm.weight"])
        k_heads = rms_norm(k_heads, w["attn.k_norm.weight"])
        if rank == 0:
            print(f"  QK norm: q_norm={q_heads.norm():.2f} k_norm={k_heads.norm():.2f}", flush=True)

    # FMHA
    from mate.jit.attention.fmha import _fmha_fwd
    scale = _ATTN_HEAD_DIM ** -0.5
    q_4d = q_heads.unsqueeze(0)
    k_4d = k_heads.unsqueeze(0)
    v_4d = v_heads.unsqueeze(0)
    attn_out, _ = _fmha_fwd(q_4d, k_4d, v_4d, softmax_scale=scale, is_causal=True)
    attn_out = attn_out.squeeze(0)
    if rank == 0:
        print(f"  FMHA: attn_norm={attn_out.norm():.2f}", flush=True)

    # Gate + flatten
    attn_flat = attn_out.reshape(total, -1)
    gate = torch.sigmoid(gate_local)
    gated = (attn_flat * gate).contiguous()
    if rank == 0:
        print(f"  Gate: sigmoid_mean={gate.mean():.3f} gated_norm={gated.norm():.2f}", flush=True)

    # AllGather
    from muserve.config import TP_SIZE
    parts = [torch.zeros_like(gated) for _ in range(TP_SIZE)]
    dist.all_gather(parts, gated)
    gathered = torch.cat(parts, dim=-1)
    if rank == 0:
        print(f"  AllGather: gathered_norm={gathered.norm():.2f}", flush=True)

    # O projection
    w_o = w["attn.o_proj.weight"]; s_o = w.get("attn.o_proj.weight_scale_inv")
    output = fp8_linear(gathered, w_o, s_o) if s_o is not None else bf16_linear(gathered, w_o)
    if rank == 0:
        print(f"  O proj: out_norm={output.norm():.2f}", flush=True)
        print(f"  Residual: hidden={hidden.norm():.2f} + out={output.norm():.2f} = {(hidden+output).norm():.2f}", flush=True)

    barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
