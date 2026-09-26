"""Tests for the adapter layer with Top-N gate, caching, and budget awareness."""
import pytest
from src.adapters import (
    TopNGate,
    BudgetAwareAdapter,
    GraphRAGAdapter,
    EvidenceGraphAdapter,
    OutcomesAnalyticsAdapter,
    PathwayGuidelineAdapter,
    AdapterResult,
    BUDGET_FREE,
    BUDGET_GPU_LITE,
    BUDGET_EXPENSIVE,
)
from src.perf import AdapterCache, PerformanceTracker, CallMetrics, get_adapter_cache, clear_adapter_caches


class TestTopNGate:
    """Test TopNGate functionality."""

    def test_initial_state(self):
        gate = TopNGate(max_calls=5)
        assert gate.can_call()
        assert gate.remaining == 5

    def test_calls_limit(self):
        gate = TopNGate(max_calls=3)
        assert gate.can_call()
        gate.record_call()
        assert gate.can_call()
        gate.record_call()
        assert gate.can_call()
        gate.record_call()
        assert not gate.can_call()
        assert gate.remaining == 0

    def test_reset(self):
        gate = TopNGate(max_calls=2)
        gate.record_call()
        gate.record_call()
        assert not gate.can_call()
        gate.reset()
        assert gate.can_call()
        assert gate.remaining == 2


class TestAdapterCache:
    """Test AdapterCache functionality."""

    def test_basic_cache(self):
        cache = AdapterCache(max_size=10)
        assert cache.size() == 0

        cache.put("key1", "value1")
        assert cache.size() == 1

        found, value = cache.get("key1")
        assert found
        assert value == "value1"

    def test_cache_miss(self):
        cache = AdapterCache(max_size=10)
        found, value = cache.get("nonexistent")
        assert not found
        assert value is None

    def test_lru_eviction(self):
        cache = AdapterCache(max_size=2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1

        found, value = cache.get("key1")
        assert not found

        found, value = cache.get("key2")
        assert found
        assert value == "value2"

    def test_memoize_decorator(self):
        cache = AdapterCache(max_size=10)
        call_count = 0

        @cache.memoize()
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not call again

    def test_clear(self):
        cache = AdapterCache(max_size=10)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0


class TestBudgetAwareAdapter:
    """Test BudgetAwareAdapter functionality."""

    def test_initial_budget(self):
        adapter = BudgetAwareAdapter(budget_limit=100.0)
        assert adapter.budget_remaining == 100.0
        assert not adapter.should_terminate()

    def test_budget_tracking(self):
        adapter = BudgetAwareAdapter(budget_limit=100.0)
        adapter.record_cost(30.0)
        assert adapter.budget_remaining == 70.0

        adapter.record_cost(50.0)
        assert adapter.budget_remaining == 20.0

    def test_budget_exceeded(self):
        adapter = BudgetAwareAdapter(budget_limit=50.0)
        adapter.record_cost(60.0)
        assert adapter.should_terminate()
        assert adapter.budget_remaining == 0.0


class TestGraphRAGAdapter:
    """Test GraphRAGAdapter functionality."""

    def setup_method(self):
        clear_adapter_caches()
        self.adapter = GraphRAGAdapter(max_calls=3)

    def test_initial_state(self):
        assert self.adapter.gate.max_calls == 3
        assert self.adapter.gate.remaining == 3

    def test_successful_retrieval(self):
        result = self.adapter.run("test query")
        assert result.success
        assert result.data is not None
        assert "query" in result.data
        assert result.data["query"] == "test query"
        assert result.cache_hit is False

    def test_cache_hit(self):
        # First call should be cache miss
        result1 = self.adapter.run("cached query")
        assert result1.cache_hit is False

        # Second call should be cache hit
        result2 = self.adapter.run("cached query")
        assert result2.cache_hit is True
        assert result1.data == result2.data

    def test_top_n_gate(self):
        # Exhaust the gate
        for i in range(3):
            result = self.adapter.run(f"query {i}")
            assert result.success

        # Next call should fail
        result = self.adapter.run("exhausted query")
        assert not result.success
        assert "Top-N gate exceeded" in result.error

    def test_budget_tracking(self):
        self.adapter.run("query 1")
        assert self.adapter._budget_spent > 0


class TestEvidenceGraphAdapter:
    """Test EvidenceGraphAdapter functionality."""

    def setup_method(self):
        clear_adapter_caches()
        self.adapter = EvidenceGraphAdapter(max_calls=2)

    def test_successful_operation(self):
        result = self.adapter.run("link_suggestion", entity_id="test_entity")
        assert result.success
        assert result.data["operation"] == "link_suggestion"

    def test_caching(self):
        result1 = self.adapter.run("kge_embeddings", entity_id="cached_entity")
        assert result1.cache_hit is False

        result2 = self.adapter.run("kge_embeddings", entity_id="cached_entity")
        assert result2.cache_hit is True

    def test_gate_exhaustion(self):
        self.adapter.run("op1", entity_id="e1")
        self.adapter.run("op2", entity_id="e2")
        result = self.adapter.run("op3", entity_id="e3")
        assert not result.success
        assert "Top-N gate exceeded" in result.error


class TestOutcomesAnalyticsAdapter:
    """Test OutcomesAnalyticsAdapter functionality."""

    def setup_method(self):
        clear_adapter_caches()
        self.adapter = OutcomesAnalyticsAdapter(max_calls=2)

    def test_survival_analysis(self):
        result = self.adapter.run("survival_analysis", cohort_id="cohort1")
        assert result.success
        assert result.data["operation"] == "survival_analysis"

    def test_budget_awareness(self):
        self.adapter.run("cohort_definition", cohort_id="c1")
        assert self.adapter._budget_spent > 0
        assert self.adapter.budget_remaining < 60.0


class TestPathwayGuidelineAdapter:
    """Test PathwayGuidelineAdapter functionality."""

    def setup_method(self):
        clear_adapter_caches()
        self.adapter = PathwayGuidelineAdapter(max_calls=3)

    def test_guideline_representation(self):
        result = self.adapter.run("represent_guideline", guideline_id="g1")
        assert result.success
        assert result.data["operation"] == "represent_guideline"

    def test_adherence_scoring(self):
        result = self.adapter.run("score_adherence", guideline_id="g1", patient_id="p1")
        assert result.success
        assert result.data["operation"] == "score_adherence"

    def test_caching(self):
        result1 = self.adapter.run("represent_guideline", guideline_id="cached_g")
        assert result1.cache_hit is False

        result2 = self.adapter.run("represent_guideline", guideline_id="cached_g")
        assert result2.cache_hit is True


class TestAdapterResult:
    """Test AdapterResult dataclass."""

    def test_default_values(self):
        result = AdapterResult()
        assert result.success is False
        assert result.data is None
        assert result.error == ""
        assert result.cache_hit is False
        assert result.budget_tier == BUDGET_FREE

    def test_success_result(self):
        result = AdapterResult(
            success=True,
            data={"key": "value"},
            cache_hit=True,
            budget_tier=BUDGET_GPU_LITE,
            duration_ms=150.5,
        )
        assert result.success
        assert result.data == {"key": "value"}
        assert result.cache_hit
        assert result.budget_tier == BUDGET_GPU_LITE
        assert result.duration_ms == 150.5


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""

    def test_initial_state(self):
        tracker = PerformanceTracker()
        assert tracker.is_enabled
        assert tracker.report.total_calls == 0

    def test_record_call(self):
        tracker = PerformanceTracker()
        metrics = CallMetrics(
            service="test",
            duration_ms=100.0,
            memory_mb=50.0,
        )
        tracker.record_call(metrics)
        assert tracker.report.total_calls == 1
        assert tracker.report.total_duration_ms == 100.0

    def test_summary(self):
        tracker = PerformanceTracker()
        metrics = CallMetrics(
            service="test",
            duration_ms=200.0,
            memory_mb=100.0,
        )
        tracker.record_call(metrics)
        summary = tracker.report.summary()
        assert summary["total_calls"] == 1
        assert summary["total_duration_ms"] == 200.0


class TestGlobalCache:
    """Test global cache registry."""

    def test_get_or_create(self):
        cache1 = get_adapter_cache("test_cache", max_size=50)
        cache2 = get_adapter_cache("test_cache", max_size=50)
        assert cache1 is cache2

    def test_different_names(self):
        cache1 = get_adapter_cache("cache1", max_size=10)
        cache2 = get_adapter_cache("cache2", max_size=20)
        assert cache1 is not cache2

    def test_clear_all(self):
        cache1 = get_adapter_cache("clear1", max_size=10)
        cache2 = get_adapter_cache("clear2", max_size=10)
        cache1.put("key", "value")
        cache2.put("key", "value")

        clear_adapter_caches()
        assert cache1.size() == 0
        assert cache2.size() == 0
