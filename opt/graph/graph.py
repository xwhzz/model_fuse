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

    InputIndex: list[int] | None = None
    # num: int = 1

    def has_weight(self, flag: bool = True) -> bool:
        if flag:
            return (self.Type == "MatMul" or self.Type == "Gemm") and len(self.Parameters) > 0
        else:
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

    def weight_is_equal(self, node1: NodeInfo, node2: NodeInfo) -> bool:
        if len(node1.Parameters) == len(node2.Parameters):
            for i in range(len(node1.Parameters)):
                hash1 = self.name2para[node1.Parameters[i]]
                hash2 = self.name2para[node2.Parameters[i]]
                if hash1 != hash2:
                    return False
                else:
                    if self.paramter_list[hash1][node1.Parameters[i]].value.shape == self.paramter_list[hash2][node2.Parameters[i]].value.shape and \
                        np.allclose(self.paramter_list[hash1][node1.Parameters[i]].value, self.paramter_list[hash2][node2.Parameters[i]].value):
                        continue
                    else:
                        return False
            return True
        return False

        # if node1.has_weight() and node2.has_weight():
        # #     return self.name2para[node1.Parameters[0]] == self.name2para[node2.Parameters[0]]
        # # else:
        # #     return False