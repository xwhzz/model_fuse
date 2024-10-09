"""
Graph Info
"""
from dataclasses import dataclass
import numpy as np
from typing import Any
from collections import defaultdict

BATCH_SYM = "batch_size"

@dataclass
class Edge: 
    name: str
    shape: list[str|int]
    src: str = ""
    dst: str = ""
    src_index: int = -1
    dst_index: int = -1

    _cached_batch_dim: int | None = None

    @property
    def batch_dim(self):
        if self._cached_batch_dim is None:
            for idx, s in enumerate(self.shape):
                if BATCH_SYM == s:
                    self._cached_batch_dim = idx
                    break
            else:
                self._cached_batch_dim = -1
        return self._cached_batch_dim

@dataclass
class Parameter:
    name: str
    value: np.ndarray
    node: str
    _cached_hash: int | None = None

    @property
    def hash(self):
        if self._cached_hash is None:
            self._cached_hash = hash(self.value.tobytes())
        return self._cached_hash


class Node: 
    def __init__(self, name: str, type: str, inputs: list[str], outputs: list[str], parameters: list[str], constants: list[str], empty: int = 0,other: Any = None, input_index: list[int] | None = None, can_batch: bool = False, domain: str | None = None, axis: int = -1):
        self.name = name
        self.type = type
        self._inputs = [[inp] for inp in inputs]
        self._outputs = [[out] for out in outputs]
        self.parameters = parameters
        self.constants = constants
        self.other = other
        self.input_index = input_index
        self.can_batch = can_batch
        self.domain = domain
        self.fuse = False
        self.axis = axis
        self.gather_list = []
        self.empty = empty
        

    @property
    def inputs(self):
        return [item[0] for item in self._inputs if item]
    
    @property
    def outputs(self):
        return [item[0] for item in self._outputs if item]

    @property
    def total_input(self):
        inputs = [None for _ in range(len(self.input_index))]
        inp_num = len(self.inputs)
        para_num = len(self.parameters)
        for i in range(inp_num):
            inputs[self.input_index[i]] = self.inputs[i]
        for i in range(para_num):
            inputs[self.input_index[inp_num+i]] = self.parameters[i]
        assert all((inp is not None for inp in inputs)), "All input should not be None."
        return inputs
    
    @property
    def gather(self):
        assert self.type == "Route", "Only Route node can be gathered."
        return f"Gather_{self.gather_list}" 
    
    @property
    def has_weight(self):
        return len(self.parameters) > 0

    def add_input_output(self, input: list[str], output: list[str]) -> None:
        assert len(input) == len(self.inputs) and len(output) == len(self.outputs), "The number of input and output should be the same."
        for idx, inp in enumerate(input):
            self._inputs[idx].append(inp)
        for idx, out in enumerate(output):
            self._outputs[idx].append(out)

    def update(self, map):
        inputs = self.inputs
        outputs = self.outputs
        self._inputs = [[map[i]] if i in map else [i] for i in inputs]
        self._outputs = [[map[i]] if i in map else [i] for i in outputs]
        # print(self._inputs, self._outputs)
    
class Graph:
    def __init__(self):
        self.node_list: dict[str, Node] = {}
        self.edge_list: dict[str, Edge] = {}

        self.parameter_list: dict[str, Parameter] = {}
        self.parameter_hash: dict[int, list[str]] = defaultdict(list)

        self.input: list[list[str] | str] = []
        self.output: list[list[str] | str] = []

        self.constants: dict[str, np.ndarray] = {}

    def add_edge(self, edge: Edge) -> None:
        self.edge_list[edge.name] = edge
    
    def add_constant(self, name: str, value: np.ndarray) -> None:
        self.constants[name] = value

    def add_node(self, node: Node, flag: bool = True) -> None:#, edges: dict[str, Edge]):
        """
        params:
        node: 需要添加的node
        edges: 与node相关联的数据流
        
        """
        self.node_list[node.name] = node
        if not flag:
            return
        edges: dict[str, Edge] = {}
        for idx, inp in enumerate(node.inputs):
            if inp not in self.edge_list: # 输入可能为空
                continue
            # assert inp in self.edge_list, f"Input {inp} not in edge list. Check whether the edge is added."
            edge = self.edge_list[inp]
            edge.dst = node.name
            edge.dst_index = idx
            edges[inp] = edge
        for idx, out in enumerate(node.outputs):
            # print(out)
            assert out in self.edge_list, f"Output {out} not in edge list. Check whether the edge is added."
            edge = self.edge_list[out]
            edge.src = node.name
            edge.src_index = idx
            edges[out] = edge
        if all((edges[inp].batch_dim != -1 for inp in node.inputs)) \
            and all((edges[out].batch_dim != -1 for out in node.outputs)):
            node.can_batch = True        
    
    def add_parameters(self, info: Parameter) -> None:
        self.parameter_list[info.name] = info
        self.parameter_hash[info.hash].append(info.name)

    def get_edge(self, name: str) -> Edge:
        return self.edge_list[name]

    def can_batch(self, node_1: Node, node_2: Node, no_weight: bool = False) -> bool:
        if no_weight and (node_1.has_weight or node_2.has_weight):
            return False
        if node_1.type == node_2.type and node_1.other == node_2.other:
            if len(node_1.input_index) == len(node_2.input_index) \
                and len(node_1.inputs) == len(node_2.inputs) \
                and len(node_1.outputs) == len(node_2.outputs):
                for idx, inp_1 in enumerate(node_1.inputs):
                    inp_2 = node_2.inputs[idx]
                    edge_1 = self.get_edge(inp_1)
                    edge_2 = self.get_edge(inp_2)
                    # if edge_1.batch_dim == -1:
                    #     return False
                    if not (edge_1.dst_index == edge_2.dst_index and \
                        edge_1.shape == edge_2.shape): #and \
                        # edge_1.batch_dim != -1):
                        return False
                for idx, out_1 in enumerate(node_1.outputs):
                    out_2 = node_2.outputs[idx]
                    edge_1 = self.get_edge(out_1)
                    edge_2 = self.get_edge(out_2)
                    # if edge_1.batch_dim == -1:
                    #     return False
                    if not (edge_1.src_index == edge_2.src_index and \
                        edge_1.shape == edge_2.shape): # and \
                        #edge_1.batch_dim != -1):
                        return False
                
                ## 相当于判断属性是否相等
                for idx, cons_1 in enumerate(node_1.constants):
                    cons_2 = node_2.constants[idx]
                    if not np.array_equal(self.constants[cons_1], self.constants[cons_2]):
                        return False
                return True
            
            else:
                return False
        return False