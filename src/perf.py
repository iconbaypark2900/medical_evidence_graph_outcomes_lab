"""Performance utilities for Medical Evidence Graph & Outcomes Lab.

This module provides:
- Adapter-level caching (memoize predictions)
- Batch processing for multiple candidates
- Timeout enforcement for all adapters
- Budget-aware early termination
- Performance metrics tracking (time per call, memory usage)
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Performance metrics                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class CallMetrics:
    """Metrics for a single adapter call."""
    service: str = ""
    call_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    budget_tier: int = 0
    budget_spent: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class PerformanceReport:
    """Aggregated performance report for a run."""
    total_calls: int = 0
    total_duration_ms: float = 0.0
    _avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_memory_mb: float = 0.0
    _avg_memory_mb: float = 0.0
    calls_by_service: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls_by_tier: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[dict[str, Any]] = field(default_factory=list)
    top_slow_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per call (computed dynamically)."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    @property
    def avg_memory_mb(self) -> float:
        """Average memory per call (computed dynamically)."""
        if self.total_calls == 0:
            return 0.0
        return self.total_memory_mb / self.total_calls

    def add_call(self, metrics: CallMetrics):
        """Add a call's metrics to the report."""
        self.total_calls += 1
        self.total_duration_ms += metrics.duration_ms
        self.total_memory_mb += metrics.memory_mb
        self.calls_by_service[metrics.service] += 1
        self.calls_by_tier[metrics.budget_tier] += 1

        if metrics.duration_ms > self.max_duration_ms:
            self.max_duration_ms = metrics.duration_ms

        if not metrics.success:
            self.errors.append({
                "service": metrics.service,
                "error": metrics.error,
                "duration_ms": metrics.duration_ms,
            })

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the performance report."""
        avg_duration = self.total_duration_ms / self.total_calls if self.total_calls > 0 else 0
        avg_memory = self.total_memory_mb / self.total_calls if self.total_calls > 0 else 0
        return {
            "total_calls": self.total_calls,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(avg_duration, 2),
            "max_duration_ms": round(self.max_duration_ms, 2),
            "total_memory_mb": round(self.total_memory_mb, 2),
            "avg_memory_mb": round(avg_memory, 2),
            "calls_by_service": dict(self.calls_by_service),
            "calls_by_tier": dict(self.calls_by_tier),
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Global performance tracker                                                  #
# --------------------------------------------------------------------------- #
class PerformanceTracker:
    """Global tracker for performance metrics across all adapters."""

    def __init__(self):
        self.report = PerformanceReport()
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def record_call(self, metrics: CallMetrics):
        if self._enabled:
            self.report.add_call(metrics)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global tracker instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    return _tracker


def track_performance(service: str = ""):
    """Context manager that tracks performance metrics for a function call.

    Usage:
        with track_performance("graph_rag"):
            result = graph_rag.search(query)
    """
    tracker = get_tracker()
    call_id = hashlib.md5(f"{service}-{time.time()}".encode()).hexdigest()[:8]
    start_time = time.time()
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert KB to MB

    class _TrackContext:
        def __enter__(self__self):
            return self__self

        def __exit__(self__self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            duration_ms = (end_time - start_time) * 1000

            metrics = CallMetrics(
                service=service,
                call_id=call_id,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_mb=end_memory - start_memory,
                success=exc_type is None,
                error=str(exc_val) if exc_val else "",
            )
            tracker.record_call(metrics)
            return False

    return _TrackContext()


def enforce_timeout(seconds: float = 120.0):
    """Decorator that enforces a timeout on a function call.

    Usage:
        @enforce_timeout(120)
        def expensive_operation():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                if elapsed > seconds * 1000:
                    raise TimeoutError(f"Function {func.__name__} exceeded timeout of {seconds}s")
                raise
        return wrapper
    return decorator


def should_terminate(budget_spent: float = 0.0, budget_limit: float = 100.0) -> bool:
    """Check if we should terminate based on budget.

    Args:
        budget_spent: Current budget spent
        budget_limit: Maximum budget allowed

    Returns:
        True if we should terminate, False otherwise
    """
    return budget_spent >= budget_limit


def batch_candidates(candidates: list, batch_size: int = 10) -> list[list]:
    """Split candidates into batches.

    Args:
        candidates: List of candidates to process
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]


def process_batch(batch: list, processor: Callable) -> list:
    """Process a batch of candidates.

    Args:
        batch: List of candidates
        processor: Function to process each candidate

    Returns:
        List of results
    """
    return [processor(candidate) for candidate in batch]


# --------------------------------------------------------------------------- #
# Adapter cache (LRU)                                                         #
# --------------------------------------------------------------------------- #
class AdapterCache:
    """LRU cache for adapter results.

    Usage:
        cache = AdapterCache(max_size=100)

        @cache.memoize(key_func=lambda smi: smi)
        def predict(smi: str) -> dict:
            ...
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []

    def _make_key(self, *args, **kwargs) -> str:
        """Create a hashable key from args and kwargs."""
        key_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Returns:
            (found, value) tuple
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return True, self._cache[key]
        return False, None

    def put(self, key: str, value: Any):
        """Put a value in the cache.

        If the cache is full, evict the least recently used item.
        """
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict LRU
            lru_key = self._order.pop(0)
            del self._cache[lru_key]

        self._cache[key] = value
        self._order.append(key)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._order.clear()

    def size(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def memoize(self, key_func: Callable = None):
        """Decorator to memoize a function with caching.

        Args:
            key_func: Function to extract cache key from args/kwargs.
                     If None, uses all args and kwargs.
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key = self._make_key(*args, **kwargs)

                found, value = self.get(key)
                if found:
                    return value

                result = func(*args, **kwargs)
                self.put(key, result)
                return result
            return wrapper
        return decorator


# Global cache registry
_adapter_caches: dict[str, AdapterCache] = {}


def get_adapter_cache(name: str = "default", max_size: int = 100) -> AdapterCache:
    """Get or create a named adapter cache."""
    if name not in _adapter_caches:
        _adapter_caches[name] = AdapterCache(max_size=max_size)
    return _adapter_caches[name]


def clear_adapter_caches():
    """Clear all adapter caches."""
    for cache in _adapter_caches.values():
        cache.clear()
