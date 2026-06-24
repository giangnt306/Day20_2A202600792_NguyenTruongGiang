"""LangGraph workflow wiring the supervisor + worker agents.

Orchestration lives here; agent internals stay in `agents/`. The supervisor is the
single router: every worker returns control to it, and it decides the next hop or stop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ANALYST,
    DONE,
    RESEARCHER,
    WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        enable_critic: bool = True,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.enable_critic = enable_critic
        self.supervisor = SupervisorAgent(self.settings)
        self.researcher = ResearcherAgent(llm=llm, search=search)
        self.analyst = AnalystAgent(llm=llm)
        self.writer = WriterAgent(llm=llm)
        self.critic = CriticAgent(llm=llm)
        self._graph: Any = None

    def _node(self, agent_attr: str) -> Callable[[ResearchState], ResearchState]:
        """Wrap an agent's run in a trace span as a LangGraph node."""

        agent: BaseAgent = getattr(self, agent_attr)

        def node(state: ResearchState) -> ResearchState:
            with trace_span(f"node:{agent.name}"):
                return agent.run(state)

        return node

    def _route(self, state: ResearchState) -> str:
        """Conditional edge: the supervisor's last decision picks the next node."""

        route = state.route_history[-1] if state.route_history else DONE
        return END if route == DONE else route

    def build(self) -> Any:
        """Create and compile the LangGraph graph."""

        graph: Any = StateGraph(ResearchState)
        graph.add_node("supervisor", self._node("supervisor"))
        graph.add_node(RESEARCHER, self._node("researcher"))
        graph.add_node(ANALYST, self._node("analyst"))
        graph.add_node(WRITER, self._node("writer"))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {RESEARCHER: RESEARCHER, ANALYST: ANALYST, WRITER: WRITER, END: END},
        )
        # Every worker reports back to the supervisor.
        for worker in (RESEARCHER, ANALYST, WRITER):
            graph.add_edge(worker, "supervisor")

        self._graph = graph.compile()
        return self._graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph, run the optional critic, and return final state."""

        graph = self._graph or self.build()
        recursion_limit = self.settings.max_iterations * 2 + 4
        with trace_span("workflow", {"query": state.request.query}):
            result = graph.invoke(state, config={"recursion_limit": recursion_limit})

        # LangGraph may return a dict or a model depending on version; normalize.
        final = (
            result
            if isinstance(result, ResearchState)
            else ResearchState.model_validate(result)
        )

        if self.enable_critic and final.final_answer:
            with trace_span("node:critic"):
                final = self.critic.run(final)
        return final
