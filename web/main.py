"""Talkie site: circa-1931 chat shell + optional proxy to GPU inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

logger = logging.getLogger(__name__)

UPSTREAM = "https://github.com/talkie-lm/talkie"
FORK = "https://github.com/CastaliaInstitute/talkie"
BLOG = "https://talkie-lm.com/"

app = FastAPI(title="Talkie (Castalia)", version="0.2.1")

# Avoid 405 on OPTIONS (CORS preflight) when the page is loaded cross-origin or probes send OPTIONS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

_CHAT_HTML: str | None = None


def _chat_html() -> str:
    global _CHAT_HTML
    if _CHAT_HTML is None:
        raw = (Path(__file__).resolve().parent / "chat_page.html").read_text(encoding="utf-8")
        _CHAT_HTML = (
            raw.replace("{{FORK}}", FORK)
            .replace("{{UPSTREAM}}", UPSTREAM)
            .replace("{{BLOG}}", BLOG)
            .replace("__TALKIE_CHAT_API_BASE__", "")
        )
    return _CHAT_HTML


def _upstream_auth_headers(target_base: str) -> dict[str, str]:
    bearer = os.environ.get("TALKIE_UPSTREAM_BEARER", "").strip()
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        req = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(req, target_base)
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        logger.warning("Upstream auth unavailable: %s", e)
        return {}


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.head("/health")
def health_head() -> Response:
    return Response(status_code=200)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _chat_html()


@app.head("/")
def index_head() -> Response:
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request) -> Response:
    """Forward OpenAI-shaped chat to GPU Cloud Run when TALKIE_UPSTREAM_URL is set."""
    base = os.environ.get("TALKIE_UPSTREAM_URL", "").strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=503,
            detail=(
                "The apparatus is not yet connected. Set TALKIE_UPSTREAM_URL on this service "
                "to your GPU Cloud Run URL (see Castalia deploy notes)."
            ),
        )

    body = await request.body()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_h = _upstream_auth_headers(base)
    if auth_h:
        headers.update(auth_h)
    else:
        # Public GPU (--no-invoker-iam-check) needs no Bearer; IAM-protected GPUs need ADC or TALKIE_UPSTREAM_BEARER.
        logger.info("Forwarding chat to upstream without Authorization header")

    url = f"{base}/v1/chat/completions"
    timeout = httpx.Timeout(connect=60.0, read=3600.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, content=body, headers=headers)
        except httpx.RequestError as e:
            logger.exception("Upstream request failed")
            raise HTTPException(status_code=502, detail=f"The line failed: {e!s}") from e

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )
