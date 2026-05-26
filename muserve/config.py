# 硬编码配置 — 只适用于 Qwen3.5-397B-A17B-FP8 + 8×S5000
# 不做抽象，不支持其他模型或硬件

# ── 模型架构 ──────────────────────────────────────────────────────────────────
NUM_LAYERS           = 60
HIDDEN_SIZE          = 4096
NUM_Q_HEADS          = 32
NUM_KV_HEADS         = 2          # 极度 GQA
HEAD_DIM             = 256
VOCAB_SIZE           = 248320
MAX_SEQ_LEN          = 262144     # 256K context

# MoE
NUM_EXPERTS          = 512
NUM_EXPERTS_PER_TOK  = 10         # top-10 routing
MOE_INTERMEDIATE     = 1024       # per-expert intermediate size
NUM_SHARED_EXPERTS   = 0

# GatedDeltaNet (线性注意力层)
GDN_NUM_K_HEADS      = 16
GDN_NUM_V_HEADS      = 64
GDN_KEY_DIM          = 128
GDN_VALUE_DIM        = 128
GDN_CONV_KERNEL      = 4

# ── 硬件 ──────────────────────────────────────────────────────────────────────
TP_SIZE              = 8          # Tensor Parallel，固定 8 卡
DEVICE               = "musa"

# ── 服务 ──────────────────────────────────────────────────────────────────────
DEFAULT_PORT         = 8000
MAX_RUNNING_REQUESTS = 256
MAX_DECODE_BATCH     = 256

# ── KV Cache ──────────────────────────────────────────────────────────────────
KV_PAGE_SIZE         = 16         # tokens per page
KV_DTYPE             = "float16"  # KV cache dtype

# ── Prefill ───────────────────────────────────────────────────────────────────
CHUNKED_PREFILL_SIZE = 4096       # chunk size for prefill（目标 TTFT 1.5s 需要大 chunk）

# ── tilelang kernel 缓存目录 ──────────────────────────────────────────────────
TILELANG_CACHE_DIR   = "/tmp/muserve_kernel_cache"

# ── 模型路径（容器内）────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH   = "/data/models/qwen3.5fp8"
