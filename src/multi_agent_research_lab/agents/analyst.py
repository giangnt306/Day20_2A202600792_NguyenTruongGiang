"""Analyst agent: turns research notes into structured, critical insights."""

from __future__ import annotations

from multi_agent_research_lab.agents.base import LLMAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

SYSTEM_PROMPT = (
    "You are a critical analyst. From the research notes, extract key claims, compare "
    "viewpoints, and flag weak or unsupported evidence. Be specific and structured. "
    "Preserve the [n] citation markers used in the notes."
)


class AnalystAgent(LLMAgent):
    """Turns research notes into structured insights."""

    agent_name = AgentName.ANALYST
    temperature = 0.1

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        notes = state.research_notes or "(no research notes available)"
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{notes}\n\n"
            "Produce: 1) Key claims (bulleted), 2) Agreements/contradictions across sources, "
            "3) Evidence gaps or weak claims to caveat."
        )
        response = self._complete(SYSTEM_PROMPT, user_prompt)
        self._record(state, response)
        state.analysis_notes = response.content
        return state
