from __future__ import annotations

from abc import ABC, abstractmethod


class DecisionPolicy(ABC):
    @abstractmethod
    def should_verify(self, action: str, step: int, trajectory: str) -> bool:
        ...

    @abstractmethod
    def which_layer(self, action: str, step: int, trajectory: str) -> int:
        ...
