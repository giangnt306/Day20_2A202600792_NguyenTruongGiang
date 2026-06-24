"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics], query: str | None = None) -> str:
    """Render benchmark metrics to a markdown comparison table."""

    lines = ["# Benchmark Report", ""]
    if query:
        lines += [f"**Query:** {query}", ""]
    lines += [
        "| Run | Latency (s) | Cost (USD) | Tokens | Quality (0-10) | Citations | Errors | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {item.total_tokens} "
            f"| {quality} | {coverage} | {item.error_count} | {item.notes} |"
        )

    lines += ["", _render_takeaways(metrics)]
    return "\n".join(lines) + "\n"


def _render_takeaways(metrics: list[BenchmarkMetrics]) -> str:
    """A short auto-generated comparison summary."""

    if len(metrics) < 2:
        return ""
    by_quality = [m for m in metrics if m.quality_score is not None]
    parts = ["## Takeaways", ""]
    fastest = min(metrics, key=lambda m: m.latency_seconds)
    cheapest = min(
        (m for m in metrics if m.estimated_cost_usd is not None),
        key=lambda m: m.estimated_cost_usd or 0.0,
        default=None,
    )
    parts.append(f"- **Fastest:** {fastest.run_name} ({fastest.latency_seconds:.2f}s)")
    if cheapest is not None:
        parts.append(
            f"- **Cheapest:** {cheapest.run_name} (${cheapest.estimated_cost_usd:.4f})"
        )
    if by_quality:
        best = max(by_quality, key=lambda m: m.quality_score or 0.0)
        parts.append(f"- **Highest quality:** {best.run_name} ({best.quality_score:.1f}/10)")
    return "\n".join(parts)
