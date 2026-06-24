"""Writer agent: synthesizes the final, cited answer for the target audience."""

from __future__ import annotations

from multi_agent_research_lab.agents.base import LLMAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

SYSTEM_PROMPT = (
    "You are a clear technical writer. Using the research and analysis notes, write a "
    "well-structured answer for the stated audience. Keep inline [n] citations and end "
    "with a 'Sources' list mapping [n] to titles/URLs. Do not introduce uncited claims."
)


class WriterAgent(LLMAgent):
    """Produces final answer from research and analysis notes."""

    agent_name = AgentName.WRITER
    temperature = 0.4

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        sources_block = "\n".join(
            f"[{i + 1}] {s.title} — {s.url or 'no-url'}" for i, s in enumerate(state.sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or '(none)'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or '(none)'}\n\n"
            f"Source map:\n{sources_block or '(none)'}\n\n"
            "Write the final answer (~500 words) with inline [n] citations and a Sources section."
        )
        response = self._complete(SYSTEM_PROMPT, user_prompt, max_tokens=1500)
        self._record(state, response)
        state.final_answer = response.content
        return state
