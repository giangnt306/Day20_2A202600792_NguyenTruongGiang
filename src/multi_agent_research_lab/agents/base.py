"""Base agent contract and shared LLM plumbing.

Agents depend on the `LLMClient`/`SearchClient` interfaces, never on a provider SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""


class LLMAgent(BaseAgent):
    """Base for agents that call the LLM and record a structured result."""

    agent_name: AgentName
    temperature: float = 0.2

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.agent_name.value

    def _complete(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1024
    ) -> LLMResponse:
        return self.llm.complete(
            system_prompt, user_prompt, temperature=self.temperature, max_tokens=max_tokens
        )

    def _record(self, state: ResearchState, response: LLMResponse) -> AgentResult:
        result = AgentResult(
            agent=self.agent_name,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.record_agent_result(result)
        return result
