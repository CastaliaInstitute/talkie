"""OpenAI-compatible HTTP API for real Talkie inference (GPU).

Serves POST /v1/chat/completions using :class:`talkie.generate.Talkie` and the
instruction-tuned chat path. Intended for Cloud Run with GPU or any CUDA host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from talkie.chat import Message
from talkie.generate import Talkie

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("TALKIE_MODEL_NAME", "talkie-1930-13b-it")
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("TALKIE_CACHE_DIR") or "/tmp/hf"
DEVICE = os.environ.get("TALKIE_DEVICE") or None

_talker: Talkie | None = None
_executor = ThreadPoolExecutor(max_workers=1)


def _load_model() -> Talkie:
    logger.warning("Loading Talkie model %s (cache=%s)...", MODEL_NAME, CACHE_DIR)
    return Talkie(MODEL_NAME, device=DEVICE, cache_dir=CACHE_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _talker
    loop = asyncio.get_running_loop()
    _talker = await loop.run_in_executor(_executor, _load_model)
    logger.warning("Talkie model ready.")
    yield
    _talker = None


app = FastAPI(title="Talkie GPU API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse | dict[str, str]:
    if _talker is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_NAME}


def _parse_messages(body: dict) -> list[Message] | None:
    raw = body.get("messages") or []
    out: list[Message] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append(Message(role=role, content=content.strip()))
    return out if out else None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if _talker is None:
        return JSONResponse(
            {"error": {"message": "model not loaded"}},
            status_code=503,
        )

    body = await request.json()
    if body.get("stream"):
        return JSONResponse(
            {"error": {"message": "stream=true is not supported yet"}},
            status_code=400,
        )

    messages = _parse_messages(body)
    if not messages:
        return JSONResponse(
            {
                "error": {
                    "message": "messages required (user/assistant/system with non-empty content)",
                }
            },
            status_code=400,
        )

    temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.7)
    max_tokens = int(body.get("max_tokens") if body.get("max_tokens") is not None else 512)
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    if top_p is not None:
        top_p = float(top_p)
    if top_k is not None:
        top_k = int(top_k)

    req_model = body.get("model") or MODEL_NAME

    loop = asyncio.get_running_loop()

    def _complete():
        assert _talker is not None
        return _talker.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )

    try:
        result = await loop.run_in_executor(_executor, _complete)
    except ValueError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=400)
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": now,
        "model": req_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
    }
