"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def record_agent_result(self, result: AgentResult) -> None:
        """Append an agent result and mirror it into the trace."""

        self.agent_results.append(result)
        self.add_trace_event(
            f"agent:{result.agent.value}",
            {"chars": len(result.content), **result.metadata},
        )

    @property
    def total_cost_usd(self) -> float:
        """Sum of estimated LLM cost recorded across agent results."""

        return float(sum(r.metadata.get("cost_usd", 0.0) or 0.0 for r in self.agent_results))

    @property
    def total_tokens(self) -> int:
        """Sum of input + output tokens recorded across agent results."""

        total = 0
        for r in self.agent_results:
            total += int(r.metadata.get("input_tokens", 0) or 0)
            total += int(r.metadata.get("output_tokens", 0) or 0)
        return total
