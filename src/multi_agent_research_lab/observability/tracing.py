"""Tracing hooks.

This file avoids binding tightly to one provider. It always records a lightweight
in-process span, and additionally emits a nested Langfuse observation when Langfuse
credentials are configured (langfuse SDK v3/v4 API).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_langfuse_client: Any | None = None
_langfuse_initialized = False
_last_trace_url: str | None = None


def _get_langfuse() -> Any | None:
    """Lazily build a singleton Langfuse client if credentials are present."""

    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client
    _langfuse_initialized = True
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("tracing: Langfuse enabled host=%s", settings.langfuse_host)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the run
        logger.warning("tracing: Langfuse init failed (%s); using local spans only", exc)
        _langfuse_client = None
    return _langfuse_client


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time a unit of work and mirror it to a nested Langfuse observation."""

    global _last_trace_url
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    lf = _get_langfuse()

    if lf is None:
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
        return

    # Langfuse path: start_as_current_observation nests child spans into one trace.
    obs_cm = None
    with contextlib.suppress(Exception):
        obs_cm = lf.start_as_current_observation(name=name, input=attributes or {})
    if obs_cm is None:
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
        return

    with obs_cm as obs:
        with contextlib.suppress(Exception):
            _last_trace_url = lf.get_trace_url()
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
            with contextlib.suppress(Exception):
                obs.update(metadata={"duration_seconds": span["duration_seconds"]})


def get_last_trace_url() -> str | None:
    """URL of the most recent Langfuse trace (None if Langfuse is disabled)."""

    return _last_trace_url


def flush_tracing() -> None:
    """Flush buffered spans to the provider (call before process exit)."""

    lf = _get_langfuse()
    if lf is not None:
        with contextlib.suppress(Exception):
            lf.flush()
