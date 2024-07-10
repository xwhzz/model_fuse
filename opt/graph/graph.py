"""
Graph Info

"""
from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class NodeInfo:

    Type: str

    Input: list[str]
    Output: list[str]
    Parameters: list[str]

    Other: Any

    # num: int = 1

    def has_weight(self) -> bool:
        return len(self.Parameters) > 0

@dataclass
class ParameterInfo:
    hash: int
    value: np.ndarray
    node: str

    @staticmethod
    def get_hash(value: np.ndarray):
        return int(np.sum(value)) % (2 ** 30)

    @classmethod
    def get_info(cls, value: np.ndarray, node: str):
        hash = cls.get_hash(value)
        return cls(hash, value, node)


class Graph:
    def __init__(self):
        self.node_list: dict[str, NodeInfo] = {}
        self.paramter_list: dict[int, dict[str, ParameterInfo]] = { }

        self.name2para: dict[str, int] = {}

        self.input = []
        self.output = []


    def add_node(self, node: NodeInfo, name: str):
        self.node_list[name] = node
        
    
    def add_parameters(self, info: ParameterInfo, name: str):
        if info.hash not in self.paramter_list:
            self.paramter_list[info.hash] = { }
        self.paramter_list[info.hash][name] = info

        self.name2para[name] = info.hash

