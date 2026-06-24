"""End-to-end workflow + baseline tests using offline mock providers."""

import pytest

from multi_agent_research_lab.agents.critic import citation_coverage
from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, estimate_cost
from multi_agent_research_lab.services.search_client import SearchClient


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with no provider keys -> clients fall back to deterministic mocks."""

    return Settings(OPENAI_API_KEY=None, TAVILY_API_KEY=None, MAX_ITERATIONS=6)


@pytest.fixture
def mock_llm(mock_settings: Settings) -> LLMClient:
    return LLMClient(mock_settings)


@pytest.fixture
def mock_search(mock_settings: Settings) -> SearchClient:
    return SearchClient(mock_settings)


def test_search_mock_returns_requested_count(mock_search: SearchClient) -> None:
    assert not mock_search.enabled
    docs = mock_search.search("graphrag", max_results=3)
    assert len(docs) == 3
    assert all(d.snippet for d in docs)


def test_llm_mock_is_deterministic(mock_llm: LLMClient) -> None:
    assert not mock_llm.enabled
    r1 = mock_llm.complete("sys", "task")
    r2 = mock_llm.complete("sys", "task")
    assert r1.content == r2.content
    assert r1.cost_usd == 0.0


def test_estimate_cost_known_and_unknown_model() -> None:
    assert estimate_cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert estimate_cost("nonexistent-model", 1000, 1000) is None


def test_baseline_populates_answer(mock_llm: LLMClient, mock_search: SearchClient) -> None:
    state = run_baseline("Explain GraphRAG", llm=mock_llm, search=mock_search)
    assert state.final_answer
    assert len(state.sources) == 5
    assert len(state.agent_results) == 1


def test_multi_agent_workflow_runs_full_pipeline(
    mock_settings: Settings, mock_llm: LLMClient, mock_search: SearchClient
) -> None:
    workflow = MultiAgentWorkflow(
        settings=mock_settings, enable_critic=True, llm=mock_llm, search=mock_search
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    assert result.research_notes
    assert result.analysis_notes
    assert result.final_answer
    # supervisor routed researcher -> analyst -> writer -> done
    assert result.route_history[:3] == ["researcher", "analyst", "writer"]
    # researcher, analyst, writer, critic each recorded a result
    assert len(result.agent_results) == 4


def test_citation_coverage() -> None:
    assert citation_coverage("uses [1] and [2]", 2) == 1.0
    assert citation_coverage("uses [1] only", 2) == 0.5
    assert citation_coverage("no citations", 3) == 0.0
    assert citation_coverage(None, 3) == 0.0
