"""Supervisor / router: decides the next worker and enforces stop conditions."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Route labels the supervisor can emit. "done" terminates the workflow.
RESEARCHER = "researcher"
ANALYST = "analyst"
WRITER = "writer"
DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def decide(self, state: ResearchState) -> str:
        """Pure routing policy: inspect state, return the next route label."""

        # Guardrail: never let the graph run unbounded.
        if state.iteration >= self.settings.max_iterations:
            logger.warning("supervisor: max_iterations reached (%d)", self.settings.max_iterations)
            return DONE

        if not state.research_notes:
            return RESEARCHER
        if not state.analysis_notes:
            return ANALYST
        if not state.final_answer:
            return WRITER
        return DONE

    def run(self, state: ResearchState) -> ResearchState:
        """Record the chosen route on the shared state."""

        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event("supervisor.route", {"next": route, "iteration": state.iteration})
        logger.info("supervisor: route=%s iteration=%d", route, state.iteration)
        return state
