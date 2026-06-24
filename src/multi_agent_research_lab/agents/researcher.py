"""Researcher agent: gathers sources and writes concise, cited research notes."""

from __future__ import annotations

from multi_agent_research_lab.agents.base import LLMAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM_PROMPT = (
    "You are a meticulous research agent. Given a query and a set of search results, "
    "write concise research notes grounded ONLY in the provided sources. "
    "Cite sources inline as [n] matching the numbered list. Do not invent facts."
)


class ResearcherAgent(LLMAgent):
    """Collects sources and creates concise research notes."""

    agent_name = AgentName.RESEARCHER
    temperature = 0.2

    def __init__(
        self, llm: LLMClient | None = None, search: SearchClient | None = None
    ) -> None:
        super().__init__(llm)
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self.search.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources

        numbered = "\n".join(
            f"[{i + 1}] {s.title} ({s.url or 'no-url'})\n{s.snippet}"
            for i, s in enumerate(sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{numbered or '(no sources found)'}\n\n"
            "Write 150-250 words of research notes with inline [n] citations."
        )
        response = self._complete(SYSTEM_PROMPT, user_prompt)
        self._record(state, response)
        state.research_notes = response.content
        return state
