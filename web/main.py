"""Minimal HTTP surface for Cloud Run (scale-to-zero). Not model inference."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

UPSTREAM = "https://github.com/talkie-lm/talkie"
FORK = "https://github.com/CastaliaInstitute/talkie"
BLOG = "https://talkie-lm.com/"

app = FastAPI(title="Talkie (Castalia)", version="0.1.0")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Talkie · Castalia Institute</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }}
    a {{ color: #0b57d0; }}
    code {{ background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Talkie</h1>
  <p>Vintage-style 13B language models (library + CLI). This host is a
  <strong>lightweight landing</strong> on Google Cloud Run (scales to zero when idle).</p>
  <p><strong>Model inference</strong> is not served here: the project needs a CUDA GPU
  with roughly 28&nbsp;GB VRAM and runs via Python locally. See the README on GitHub.</p>
  <ul>
    <li><a href="{FORK}">Castalia Institute fork</a></li>
    <li><a href="{UPSTREAM}">Upstream repository</a></li>
    <li><a href="{BLOG}">Project blog</a></li>
  </ul>
  <p><a href="/health"><code>/health</code></a> for probes.</p>
</body>
</html>"""

