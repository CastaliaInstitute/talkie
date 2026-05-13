"""Coarse-grained model load progress for HTTP health / UI meters (thread-safe)."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_phase: str = "idle"
_label: str = ""
_step: int | None = None
_total: int | None = None


def reset(phase: str = "starting", label: str = "", *, step: int | None = None, total: int | None = None) -> None:
    """Call at the beginning of a load (e.g. :class:`Talkie` construction)."""

    global _phase, _label, _step, _total
    with _lock:
        _phase = phase
        _label = label
        _step = step
        _total = total


def update(
    phase: str,
    label: str,
    *,
    step: int | None = None,
    total: int | None = None,
) -> None:
    """Publish the current load stage; optional *step* / *total* for determinate meters."""

    global _phase, _label, _step, _total
    with _lock:
        _phase = phase
        _label = label
        _step = step
        _total = total


def mark_ready() -> None:
    """Call when weights are fully loaded and the model is ready to run."""

    global _phase, _label, _step, _total
    with _lock:
        _phase = "ready"
        _label = "Model ready"
        _step = 1
        _total = 1


def snapshot() -> dict[str, Any]:
    """Return a JSON-serializable progress dict."""

    with _lock:
        pct: float | None = None
        if _step is not None and _total is not None and _total > 0:
            pct = min(1.0, max(0.0, _step / _total))
        return {
            "phase": _phase,
            "label": _label,
            "step": _step,
            "total": _total,
            "pct": pct,
        }
