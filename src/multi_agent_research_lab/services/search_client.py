"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when an API key is configured, otherwise returns deterministic mock
    documents so the workflow stays runnable offline.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        if self.settings.search_enabled:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.settings.tavily_api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        max_results = max(1, min(max_results, 20))
        if self._client is None:
            return self._mock_search(query, max_results)
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on provider failure
            logger.warning("search.tavily failed (%s); falling back to mock", exc)
            return self._mock_search(query, max_results)

        documents: list[SourceDocument] = []
        for item in response.get("results", [])[:max_results]:
            documents.append(
                SourceDocument(
                    title=item.get("title") or "Untitled",
                    url=item.get("url"),
                    snippet=(item.get("content") or "").strip(),
                    metadata={"score": item.get("score")},
                )
            )
        logger.info("search.tavily query=%r results=%d", query, len(documents))
        return documents

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        logger.warning("search using MOCK provider (no TAVILY_API_KEY configured)")
        return [
            SourceDocument(
                title=f"Mock source {i + 1} for: {query[:50]}",
                url=f"https://example.com/mock/{i + 1}",
                snippet=(
                    f"Mock evidence #{i + 1} relevant to '{query}'. "
                    "Configure TAVILY_API_KEY in .env for real search results."
                ),
                metadata={"mock": True, "rank": i + 1},
            )
            for i in range(max_results)
        ]
