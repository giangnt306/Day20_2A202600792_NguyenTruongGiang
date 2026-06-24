"""Routing-policy tests for the supervisor (now implemented, mock-friendly)."""

from multi_agent_research_lab.agents.supervisor import (
    ANALYST,
    DONE,
    RESEARCHER,
    WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    assert SupervisorAgent().decide(_state()) == RESEARCHER


def test_supervisor_progresses_through_pipeline() -> None:
    supervisor = SupervisorAgent()
    state = _state()

    state.research_notes = "notes"
    assert supervisor.decide(state) == ANALYST

    state.analysis_notes = "analysis"
    assert supervisor.decide(state) == WRITER

    state.final_answer = "answer"
    assert supervisor.decide(state) == DONE


def test_supervisor_stops_at_max_iterations() -> None:
    supervisor = SupervisorAgent(Settings(MAX_ITERATIONS=2))
    state = _state()
    state.iteration = 2
    assert supervisor.decide(state) == DONE


def test_supervisor_run_records_route() -> None:
    state = _state()
    SupervisorAgent().run(state)
    assert state.route_history == [RESEARCHER]
    assert state.iteration == 1
