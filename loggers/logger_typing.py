from typing import Protocol, Dict


class Logger(Protocol):
    def log(self, metrics: Dict): ...
