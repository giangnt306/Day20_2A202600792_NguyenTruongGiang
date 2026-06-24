"""Optional critic agent: checks citation coverage and flags unsupported claims."""

from __future__ import annotations

import re

from multi_agent_research_lab.agents.base import LLMAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

SYSTEM_PROMPT = (
    "You are a fact-checking critic. Review the final answer against the sources. "
    "Identify any claims that lack a citation or are not supported by the sources, and "
    "rate overall citation coverage. Be concise and actionable."
)


def citation_coverage(answer: str | None, num_sources: int) -> float:
    """Fraction of available sources that are actually cited as [n] in the answer."""

    if not answer or num_sources <= 0:
        return 0.0
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= num_sources}
    return len(cited) / num_sources


class CriticAgent(LLMAgent):
    """Optional fact-checking and safety-review agent."""

    agent_name = AgentName.CRITIC
    temperature = 0.0

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to state."""

        coverage = citation_coverage(state.final_answer, len(state.sources))
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Final answer:\n{state.final_answer or '(none)'}\n\n"
            f"Number of sources: {len(state.sources)}\n"
            f"Measured citation coverage: {coverage:.0%}\n\n"
            "List unsupported/uncited claims (if any) and give a one-line verdict."
        )
        response = self._complete(SYSTEM_PROMPT, user_prompt)
        result = self._record(state, response)
        result.metadata["citation_coverage"] = coverage
        if coverage < 0.5:
            state.errors.append(f"Low citation coverage: {coverage:.0%}")
        return state
