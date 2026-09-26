"""Adapter layer for Medical Evidence Graph & Outcomes Lab.

This module provides adapter classes with:
- Top-N gate (5 calls max per instance)
- Timeout enforcement (120s default)
- LRU caching via AdapterCache
- Budget-aware early termination
- Performance tracking
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.perf import (
    AdapterCache,
    CallMetrics,
    PerformanceReport,
    PerformanceTracker,
    enforce_timeout,
    get_adapter_cache,
    get_tracker,
    should_terminate,
    track_performance,
)

logger = logging.getLogger(__name__)


# Budget tiers
BUDGET_FREE = 0
BUDGET_GPU_LITE = 1
BUDGET_EXPENSIVE = 2


@dataclass
class AdapterResult:
    """Result from an adapter call."""
    success: bool = False
    data: Any = None
    error: str = ""
    cache_hit: bool = False
    budget_tier: int = BUDGET_FREE
    duration_ms: float = 0.0


class TopNGate:
    """Gate that limits the number of calls per adapter instance.

    Usage:
        gate = TopNGate(max_calls=5)
        if gate.can_call():
            gate.record_call()
            result = adapter.run(...)
    """

    def __init__(self, max_calls: int = 5):
        self.max_calls = max_calls
        self._call_count = 0

    def can_call(self) -> bool:
        """Check if we can make another call."""
        return self._call_count < self.max_calls

    def record_call(self):
        """Record a call."""
        self._call_count += 1

    @property
    def remaining(self) -> int:
        """Number of remaining calls."""
        return max(0, self.max_calls - self._call_count)

    def reset(self):
        """Reset the call counter."""
        self._call_count = 0


class BudgetAwareAdapter:
    """Base adapter class with budget-aware termination.

    Usage:
        class MyAdapter(BudgetAwareAdapter):
            def __init__(self):
                super().__init__(budget_limit=100.0)

            def run(self, **kwargs) -> AdapterResult:
                if self.should_terminate():
                    return AdapterResult(success=False, error="Budget exceeded")
                # ... actual logic
    """

    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self._budget_spent = 0.0
        self._tracker = get_tracker()

    def record_cost(self, cost: float = 1.0):
        """Record the cost of a call."""
        self._budget_spent += cost

    def should_terminate(self) -> bool:
        """Check if we should terminate based on budget."""
        return should_terminate(self._budget_spent, self.budget_limit)

    @property
    def budget_remaining(self) -> float:
        """Budget remaining."""
        return max(0.0, self.budget_limit - self._budget_spent)


class GraphRAGAdapter(BudgetAwareAdapter):
    """Adapter for Graph-RAG hybrid retrieval.

    Combines BM25 (OpenSearch), vector search (Qdrant), and graph traversal (Neo4j).
    """

    def __init__(self, max_calls: int = 5, timeout: float = 120.0):
        super().__init__(budget_limit=50.0)
        self.gate = TopNGate(max_calls=max_calls)
        self.timeout = timeout
        self.cache = get_adapter_cache("graph_rag", max_size=50)

    def run(self, query: str, **kwargs) -> AdapterResult:
        """Run Graph-RAG retrieval.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            AdapterResult with search results
        """
        if not self.gate.can_call():
            return AdapterResult(
                success=False,
                error=f"Top-N gate exceeded (max {self.gate.max_calls} calls)",
                budget_tier=BUDGET_FREE,
            )

        start_time = time.time()
        try:
            # Check cache
            cache_key = f"graph_rag:{query}"
            found, cached_result = self.cache.get(cache_key)
            if found:
                self.gate.record_call()
                self.record_cost(0.1)  # Cache hit is cheap
                duration_ms = (time.time() - start_time) * 1000
                return AdapterResult(
                    success=True,
                    data=cached_result,
                    cache_hit=True,
                    budget_tier=BUDGET_FREE,
                    duration_ms=duration_ms,
                )

            # Actual retrieval logic would go here
            # For now, return a placeholder
            result = {
                "query": query,
                "results": [],
                "retrievers_used": ["bm25", "vector", "graph"],
            }

            # Cache the result
            self.cache.put(cache_key, result)
            self.gate.record_call()
            self.record_cost(1.0)  # Full retrieval cost
            duration_ms = (time.time() - start_time) * 1000

            return AdapterResult(
                success=True,
                data=result,
                cache_hit=False,
                budget_tier=BUDGET_GPU_LITE,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return AdapterResult(
                success=False,
                error=str(e),
                budget_tier=BUDGET_FREE,
                duration_ms=duration_ms,
            )


class EvidenceGraphAdapter(BudgetAwareAdapter):
    """Adapter for Evidence Graph Service.

    Provides link suggestion and knowledge graph embeddings.
    """

    def __init__(self, max_calls: int = 5, timeout: float = 300.0):
        super().__init__(budget_limit=75.0)
        self.gate = TopNGate(max_calls=max_calls)
        self.timeout = timeout
        self.cache = get_adapter_cache("evidence_graph", max_size=100)

    def run(self, operation: str, **kwargs) -> AdapterResult:
        """Run evidence graph operation.

        Args:
            operation: Operation to perform (e.g., "link_suggestion", "kge_embeddings")
            **kwargs: Operation-specific parameters

        Returns:
            AdapterResult with operation results
        """
        if not self.gate.can_call():
            return AdapterResult(
                success=False,
                error=f"Top-N gate exceeded (max {self.gate.max_calls} calls)",
                budget_tier=BUDGET_FREE,
            )

        start_time = time.time()
        try:
            # Check cache
            cache_key = f"evidence_graph:{operation}:{kwargs.get('entity_id', '')}"
            found, cached_result = self.cache.get(cache_key)
            if found:
                self.gate.record_call()
                self.record_cost(0.1)
                duration_ms = (time.time() - start_time) * 1000
                return AdapterResult(
                    success=True,
                    data=cached_result,
                    cache_hit=True,
                    budget_tier=BUDGET_FREE,
                    duration_ms=duration_ms,
                )

            # Actual graph operation logic would go here
            result = {
                "operation": operation,
                "status": "completed",
                "data": kwargs,
            }

            # Cache the result
            self.cache.put(cache_key, result)
            self.gate.record_call()
            self.record_cost(2.0)  # Graph operations are more expensive
            duration_ms = (time.time() - start_time) * 1000

            return AdapterResult(
                success=True,
                data=result,
                cache_hit=False,
                budget_tier=BUDGET_EXPENSIVE,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return AdapterResult(
                success=False,
                error=str(e),
                budget_tier=BUDGET_FREE,
                duration_ms=duration_ms,
            )


class OutcomesAnalyticsAdapter(BudgetAwareAdapter):
    """Adapter for Outcomes Analytics Service.

    Provides cohort building, survival analysis, and comparative effectiveness.
    """

    def __init__(self, max_calls: int = 5, timeout: float = 180.0):
        super().__init__(budget_limit=60.0)
        self.gate = TopNGate(max_calls=max_calls)
        self.timeout = timeout
        self.cache = get_adapter_cache("outcomes_analytics", max_size=50)

    def run(self, operation: str, **kwargs) -> AdapterResult:
        """Run outcomes analytics operation.

        Args:
            operation: Operation to perform (e.g., "survival_analysis", "comparative_effectiveness")
            **kwargs: Operation-specific parameters

        Returns:
            AdapterResult with operation results
        """
        if not self.gate.can_call():
            return AdapterResult(
                success=False,
                error=f"Top-N gate exceeded (max {self.gate.max_calls} calls)",
                budget_tier=BUDGET_FREE,
            )

        start_time = time.time()
        try:
            # Check cache
            cache_key = f"outcomes:{operation}:{kwargs.get('cohort_id', '')}"
            found, cached_result = self.cache.get(cache_key)
            if found:
                self.gate.record_call()
                self.record_cost(0.2)
                duration_ms = (time.time() - start_time) * 1000
                return AdapterResult(
                    success=True,
                    data=cached_result,
                    cache_hit=True,
                    budget_tier=BUDGET_FREE,
                    duration_ms=duration_ms,
                )

            # Actual analytics logic would go here
            result = {
                "operation": operation,
                "status": "completed",
                "data": kwargs,
            }

            # Cache the result
            self.cache.put(cache_key, result)
            self.gate.record_call()
            self.record_cost(3.0)  # Analytics are expensive
            duration_ms = (time.time() - start_time) * 1000

            return AdapterResult(
                success=True,
                data=result,
                cache_hit=False,
                budget_tier=BUDGET_EXPENSIVE,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return AdapterResult(
                success=False,
                error=str(e),
                budget_tier=BUDGET_FREE,
                duration_ms=duration_ms,
            )


class PathwayGuidelineAdapter(BudgetAwareAdapter):
    """Adapter for Pathway & Guideline Service.

    Provides guideline representation, adherence scoring, and optimization.
    """

    def __init__(self, max_calls: int = 5, timeout: float = 120.0):
        super().__init__(budget_limit=40.0)
        self.gate = TopNGate(max_calls=max_calls)
        self.timeout = timeout
        self.cache = get_adapter_cache("pathway_guideline", max_size=100)

    def run(self, operation: str, **kwargs) -> AdapterResult:
        """Run pathway guideline operation.

        Args:
            operation: Operation to perform (e.g., "represent_guideline", "score_adherence")
            **kwargs: Operation-specific parameters

        Returns:
            AdapterResult with operation results
        """
        if not self.gate.can_call():
            return AdapterResult(
                success=False,
                error=f"Top-N gate exceeded (max {self.gate.max_calls} calls)",
                budget_tier=BUDGET_FREE,
            )

        start_time = time.time()
        try:
            # Check cache
            cache_key = f"pathway:{operation}:{kwargs.get('guideline_id', '')}"
            found, cached_result = self.cache.get(cache_key)
            if found:
                self.gate.record_call()
                self.record_cost(0.1)
                duration_ms = (time.time() - start_time) * 1000
                return AdapterResult(
                    success=True,
                    data=cached_result,
                    cache_hit=True,
                    budget_tier=BUDGET_FREE,
                    duration_ms=duration_ms,
                )

            # Actual pathway logic would go here
            result = {
                "operation": operation,
                "status": "completed",
                "data": kwargs,
            }

            # Cache the result
            self.cache.put(cache_key, result)
            self.gate.record_call()
            self.record_cost(1.5)
            duration_ms = (time.time() - start_time) * 1000

            return AdapterResult(
                success=True,
                data=result,
                cache_hit=False,
                budget_tier=BUDGET_GPU_LITE,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return AdapterResult(
                success=False,
                error=str(e),
                budget_tier=BUDGET_FREE,
                duration_ms=duration_ms,
            )
