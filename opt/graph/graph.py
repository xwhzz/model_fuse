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

    def has_weight(self, flag: bool = True, graph = None) -> bool:
        if flag and graph is not None:
            has_weight = False
            for para in self.Parameters:
                if para not in graph.constants:
                    has_weight = True
                    break
            return has_weight # or (self.Type == "MatMul" and len(self.Parameters) > 0 )
            # return (self.Type == "MatMul" or self.Type == "Gemm") and len(self.Parameters) > 0
        else:
            return len(self.Parameters) > 0
    
    def can_batch(self, name2shape, input_default: int=1) -> bool:
        ## 可以扩展到多个输入的情况
        # input_flag = True
        # for inp in self.Input[:input_default]:
        for inp in self.Input:
            if not name2shape[inp]:
                # input_flag = False
                # break
                return False
        for out in self.Output:
            if not name2shape[out]:
                return False
        return True
        # return len(self.Output) == 1 and name2shape[self.Output[0]] and input_flag and len(self.Input) == input_default
    
@dataclass
class ParameterInfo:
    hash: int
    value: np.ndarray
    node: str

    @staticmethod
    def get_hash(value: np.ndarray):
        # Use more robust hash function
        return hash(value.tobytes()) 

    @classmethod
    def get_info(cls, value: np.ndarray, node: str):
        hash = cls.get_hash(value)
        return cls(hash, value, node)


class Graph:
    def __init__(self):
        self.node_list: dict[str, NodeInfo] = {}
        self.paramter_list: dict[int, dict[str, ParameterInfo]] = { }

        self.name2para: dict[str, int] = {}
        # extend to support multiple inputs and outputs
        # FIXME: support inputs with no batch size
        self.input: list[list[str] | str] = []
        self.output: list[list[str] | str] = []

        self.name2shape: dict[str, bool] = {}

        self.constants: set[str] = set()

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

    def weight_is_equal(self, node1: NodeInfo, node2: NodeInfo, node1_index: int=0, node2_index: int=0) -> bool:
        if node1.Type == node2.Type and len(node1.Parameters) == len(node2.Parameters) and node1_index == node2_index:
            # try:
            #     for inp in node1.Input:
            #         if not self.name2shape[inp]:
            #             return False
            #     for out in node1.Output:
            #         if not self.name2shape[out]:
            #             return False
            #     for inp in node2.Input:
            #         if not self.name2shape[inp]:
            #             return False
            #     for out in node2.Output:
            #         if not self.name2shape[out]:
            #             return False
            # except:
            #     pass
            if not node1.Can_batch:
                return False
            if not node2.Can_batch:
                return False
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
            if node1.Other != node2.Other:
                return False
            return True
        return False

    def add_constant(self, name: str):
        self.constants.add(name)
        # if node1.has_weight() and node2.has_weight():
        # #     return self.name2para[node1.Parameters[0]] == self.name2para[node2.Parameters[0]]
        # # else:
        # #     return False