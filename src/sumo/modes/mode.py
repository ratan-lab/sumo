from abc import ABC, abstractmethod


class SumoMode(ABC):
    """Defines modes of sumo"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self):
        """Run mode specific functionality"""
        raise NotImplementedError("Not implemented")
