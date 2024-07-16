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

    Can_batch: bool | None = None
    # num: int = 1

    def has_weight(self) -> bool:
        return len(self.Parameters) > 0
    
    def can_batch(self, name2shape) -> bool:
        return len(self.Input) == 1 and len(self.Output) == 1 and name2shape[self.Input[0]] and name2shape[self.Output[0]]
    


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

        # 我们需要得到每个数据的第一个维度是什么？
        self.name2shape: dict[str, bool] = {}

    def add_node(self, node: NodeInfo, name: str):
        self.node_list[name] = node
        try:
            node.Can_batch = node.can_batch(self.name2shape)
        except:
            pass
    
    def add_parameters(self, info: ParameterInfo, name: str):
        if info.hash not in self.paramter_list:
            self.paramter_list[info.hash] = { }
        self.paramter_list[info.hash][name] = info

        self.name2para[name] = info.hash

