"""
Base.py

"""
from abc import ABC, abstractmethod

class Converter(ABC):
    @abstractmethod
    def to_graph(self, model):
        pass

    @abstractmethod
    def from_graph(self, graph):
        pass