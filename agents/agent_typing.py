import numpy as np

from typing import Protocol


class Agent(Protocol):
    def act(self, observation: np.ndarray) -> int: ...