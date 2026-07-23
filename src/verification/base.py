from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def check(self, action: str) -> tuple[bool, str | None]:
        """Pure predicate: must not mutate per-episode state. See `commit`."""
        ...

    def commit(self, action: str) -> None:
        """Called after the action actually ran in the environment. Verifiers that
        track history (duplicates, preconditions) record it here, so an action
        that was blocked or revised away is not remembered as having run."""
        return None

    @property
    def violation_count(self) -> int:
        return 0
