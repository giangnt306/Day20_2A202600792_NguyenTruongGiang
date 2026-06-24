"""Benchmark single-agent vs multi-agent on latency, cost, quality, and citations."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.agents.critic import citation_coverage
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_JUDGE_SYSTEM = (
    "You are a strict evaluator. Score the answer for the query on a 0-10 scale for "
    "accuracy, completeness, clarity, and grounding in citations. Reply with ONLY the "
    "number (an integer or one decimal), nothing else."
)


def judge_quality(query: str, answer: str, llm: LLMClient | None = None) -> float | None:
    """LLM-as-judge quality score in [0, 10]; None if unavailable."""

    if not answer:
        return 0.0
    llm = llm or LLMClient()
    try:
        response = llm.complete(
            _JUDGE_SYSTEM,
            f"Query: {query}\n\nAnswer:\n{answer}\n\nScore (0-10):",
            temperature=0.0,
            max_tokens=8,
        )
    except Exception as exc:  # noqa: BLE001 - judging must not crash a benchmark
        logger.warning("judge_quality failed: %s", exc)
        return None
    match = re.search(r"\d+(?:\.\d+)?", response.content)
    if not match:
        return None
    return max(0.0, min(10.0, float(match.group())))


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    judge: bool = True,
    llm: LLMClient | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner`, time it, and compute quality/cost/citation metrics."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - record failure as a metric, don't crash
        latency = perf_counter() - started
        logger.error("benchmark run %r failed: %s", run_name, exc)
        failed = ResearchState(request=ResearchQuery(query=query))
        failed.errors.append(str(exc))
        return failed, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            error_count=1,
            notes=f"run failed: {exc}",
        )

    latency = perf_counter() - started
    error_count = len(state.errors)
    coverage = citation_coverage(state.final_answer, len(state.sources))
    quality = (
        judge_quality(query, state.final_answer or "", llm=llm) if judge else None
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=round(state.total_cost_usd, 6),
        quality_score=quality,
        total_tokens=state.total_tokens,
        citation_coverage=round(coverage, 3),
        error_count=error_count,
        notes=f"agents={len(state.agent_results)}, sources={len(state.sources)}",
    )
    return state, metrics
