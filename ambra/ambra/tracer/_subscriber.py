"""Base Subscriber class."""

import logging
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Subscriber(ABC):
    """Abstract base class for trace subscribers.

    Args:
        patterns: List of regex patterns to match trace identifiers.
                  If None or empty, all traces are matched.

    """

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self._patterns: Dict[str, re.Pattern[str]] = {}
        self._active_tracepoints: Dict[str, bool] = {}
        self._active_tracepoints_lock = threading.Lock()
        self._verbose = verbose

        # Add initial patterns
        if patterns:
            for pattern in patterns:
                self.add_pattern(pattern)

    def add_pattern(self, pattern: str) -> None:
        """Add a pattern for matching trace identifiers.

        Clears the active tracepoints cache to ensure tracepoints are
        re-evaluated against the new pattern.

        Args:
            pattern: Regex pattern to add.

        """
        compiled = re.compile(pattern)
        with self._active_tracepoints_lock:
            self._patterns[pattern] = compiled
            self._active_tracepoints.clear()

        if self._verbose:
            logger.debug("Added pattern: %r", pattern)

    def remove_pattern(self, pattern: str) -> None:
        """Remove a pattern.

        Clears the active tracepoints cache to ensure tracepoints are
        re-evaluated against remaining patterns.

        Args:
            pattern: The pattern string to remove.

        """
        with self._active_tracepoints_lock:
            self._patterns.pop(pattern, None)
            self._active_tracepoints.clear()
        if self._verbose:
            logger.debug("Removed pattern: %r", pattern)

    def get_patterns(self) -> List[str]:
        """Get list of current patterns.

        Returns:
            List of pattern strings.

        """
        with self._active_tracepoints_lock:
            return list(self._patterns.keys())

    def get_active_tracepoints(self) -> Set[str]:
        """Get the set of active tracepoint identifiers.

        Returns:
            Set of identifier strings that are active.

        """
        with self._active_tracepoints_lock:
            return {k for k, v in self._active_tracepoints.items() if v}

    def check_active(self, identifier: str) -> bool:
        """Check if an identifier is active, updating cache if needed.

        Uses cached results for O(1) lookup first, falls back to pattern
        matching only on first occurrence of an identifier.
        Results are cached as True (active) or False (inactive).

        Args:
            identifier: The trace identifier to check.

        Returns:
            True if the identifier matches patterns, False otherwise.

        """
        with self._active_tracepoints_lock:
            # Fast path
            if (enabled := self._active_tracepoints.get(identifier)) is not None:
                return enabled

            # Slow path
            enabled = any(p.fullmatch(identifier) for p in self._patterns.values())
            self._active_tracepoints[identifier] = enabled
            if enabled and self._verbose:
                logger.debug("Activated tracepoint: %r", identifier)
            return enabled

    @abstractmethod
    def start(self) -> None:
        """Start the subscriber."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the subscriber."""

    @abstractmethod
    def on_trace(self, identifier: str, kwargs: Dict[str, Any]) -> None:
        """Called when a trace event occurs for an active tracepoint.

        The implementation should assume that trace data in kwargs could be
        invalidated after the function returns and is responsible for copying
        or serializing the data before returning.

        Args:
            identifier: The trace identifier.
            kwargs: The trace data.

        """
