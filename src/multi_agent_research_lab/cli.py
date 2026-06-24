"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import flush_tracing, get_last_trace_url
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    mode = "LLM=real" if settings.llm_enabled else "LLM=mock"
    search = "search=real" if settings.search_enabled else "search=mock"
    console.print(f"[dim]{mode}, {search}, model={settings.openai_model}[/dim]")


def _finish() -> None:
    """Flush traces and print the Langfuse trace link, if any."""

    flush_tracing()
    url = get_last_trace_url()
    if url:
        console.print(f"[cyan]Langfuse trace:[/cyan] {url}")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))
    console.print(
        f"[dim]cost=${state.total_cost_usd:.4f} tokens={state.total_tokens}[/dim]"
    )
    _finish()


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    no_critic: Annotated[bool, typer.Option("--no-critic", help="Disable critic step")] = False,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(enable_critic=not no_critic)
    result = workflow.run(state)

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(
        f"[dim]route={' -> '.join(result.route_history)} | "
        f"cost=${result.total_cost_usd:.4f} tokens={result.total_tokens} | "
        f"errors={len(result.errors)}[/dim]"
    )
    _finish()


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o")] = "benchmark_report.md",
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip LLM quality judge")] = False,
) -> None:
    """Benchmark single-agent vs multi-agent and write a markdown report."""

    _init()
    judge = not no_judge

    _, base_metrics = run_benchmark(
        "single-agent", query, lambda q: run_baseline(q), judge=judge
    )
    _, multi_metrics = run_benchmark(
        "multi-agent",
        query,
        lambda q: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=q))),
        judge=judge,
    )

    report = render_markdown_report([base_metrics, multi_metrics], query=query)
    path = LocalArtifactStore().write_text(output, report)
    console.print(report)
    console.print(f"[green]Report written to {path}[/green]")
    _finish()


if __name__ == "__main__":
    app()
