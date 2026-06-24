"""Single-agent baseline: one LLM call does research + analysis + writing.

This is the control arm benchmarked against the multi-agent workflow.
"""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM_PROMPT = (
    "You are a single research assistant. In one pass, use the provided sources to "
    "research, analyze, and write a clear, well-structured answer for the audience. "
    "Use inline [n] citations and end with a Sources list. Do not invent facts."
)


def run_baseline(
    query: str,
    *,
    llm: LLMClient | None = None,
    search: SearchClient | None = None,
    max_sources: int = 5,
) -> ResearchState:
    """Run the single-agent baseline and return the populated state."""

    llm = llm or LLMClient()
    search = search or SearchClient()

    state = ResearchState(request=ResearchQuery(query=query, max_sources=max_sources))
    state.sources = search.search(query, max_results=max_sources)

    numbered = "\n".join(
        f"[{i + 1}] {s.title} ({s.url or 'no-url'})\n{s.snippet}"
        for i, s in enumerate(state.sources)
    )
    user_prompt = (
        f"Query: {query}\n"
        f"Audience: {state.request.audience}\n\n"
        f"Sources:\n{numbered or '(no sources found)'}\n\n"
        "Write a ~500-word answer with inline [n] citations and a Sources section."
    )
    response = llm.complete(SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=1500)
    state.record_agent_result(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "mode": "baseline",
            },
        )
    )
    state.final_answer = response.content
    return state
