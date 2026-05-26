"""Task 1.2：OpenAI-compatible API server with SSE streaming.

用法：
    torchrun --nproc-per-node=8 muserve/server.py [--port 8000]
"""

import argparse
import json
import time
import uuid
import threading
from typing import Generator

import torch
import torch_musa
from flask import Flask, request, Response, jsonify

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_layer_weights, load_embedding_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS, TP_SIZE, VOCAB_SIZE

app = Flask(__name__)
model: Qwen35Model = None
tokenizer = None
device = None


def load_model(model_path: str):
    """加载完整模型权重。"""
    global model, device
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[server] Loading model from {model_path}...")

    weight_index = _load_index(model_path)
    embed_w = load_embedding_weights(model_path, weight_index)
    layer_ws = []
    for i in range(NUM_LAYERS):
        layer_ws.append(load_layer_weights(model_path, i, weight_index))
        if rank == 0 and (i + 1) % 10 == 0:
            used = torch.musa.memory_allocated(device) / 1e9
            print(f"  layer {i+1}/{NUM_LAYERS}, GPU mem: {used:.1f} GB")

    model = Qwen35Model(embed_w, layer_ws)
    barrier()
    if rank == 0:
        print(f"[server] Model loaded successfully")


def load_tokenizer(model_path: str):
    """加载 tokenizer。"""
    global tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if get_tp_rank() == 0:
        print(f"[server] Tokenizer loaded, vocab_size={tokenizer.vocab_size}")


def generate_stream(
    input_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Generator[str, None, None]:
    """流式生成 token，yield SSE 格式的 JSON chunk。"""
    rank = get_tp_rank()
    request_id = str(uuid.uuid4())[:8]

    ids_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, len(input_ids)], device=device, dtype=torch.int64)

    # Prefill
    logits, gdn_states = model.forward_prefill(ids_tensor, cu_seqlens)

    # Sample first token
    if temperature <= 0:
        next_token = model.greedy_sample(logits)
    else:
        next_token = model.greedy_sample(logits)  # TODO: temperature sampling

    generated = [next_token.item()]

    if rank == 0:
        token_text = tokenizer.decode([next_token.item()], skip_special_tokens=False)
        chunk = _make_chunk(request_id, token_text)
        yield f"data: {json.dumps(chunk)}\n\n"

    # Decode loop
    for step in range(max_new_tokens - 1):
        decode_ids = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)

        if temperature <= 0:
            next_token = model.greedy_sample(logits)
        else:
            next_token = model.greedy_sample(logits)

        token_id = next_token.item()
        generated.append(token_id)

        # EOS check
        if tokenizer and token_id == tokenizer.eos_token_id:
            break

        if rank == 0:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            chunk = _make_chunk(request_id, token_text)
            yield f"data: {json.dumps(chunk)}\n\n"

    if rank == 0:
        yield f"data: {json.dumps(_make_chunk(request_id, '', finish_reason='stop'))}\n\n"
        yield "data: [DONE]\n\n"


def _make_chunk(request_id: str, content: str, finish_reason: str = None) -> dict:
    """构造 OpenAI SSE chunk 格式。"""
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3.5-397b",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.0)
    stream = data.get("stream", False)

    # 用 tokenizer 编码
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt)

    if stream:
        return Response(
            generate_stream(input_ids, max_tokens, temperature),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        # Non-streaming: collect all tokens
        output_text = ""
        for chunk_str in generate_stream(input_ids, max_tokens, temperature):
            if chunk_str.startswith("data: [DONE]"):
                break
            if chunk_str.startswith("data: "):
                chunk = json.loads(chunk_str[6:])
                delta = chunk["choices"][0].get("delta", {})
                output_text += delta.get("content", "")

        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3.5-397b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
        })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [{"id": "qwen3.5-397b", "object": "model"}]
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()

    load_model(args.model_path)
    load_tokenizer(args.model_path)

    barrier()

    if rank == 0:
        print(f"[server] Starting API server on port {args.port}")
        app.run(host="0.0.0.0", port=args.port, threaded=False)
    else:
        # Non-rank-0 workers wait for inference requests via broadcast
        # In eager baseline, all ranks run the same forward pass
        # Flask only runs on rank 0, other ranks participate via dist collectives
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
