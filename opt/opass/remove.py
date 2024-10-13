from ..graph import *
from ..utils import *
from collections import defaultdict
from collections import deque

def get_map(graph: Graph):
    input2node = defaultdict(list)
    output2node = defaultdict(list)
    custom_list = {} 

    for node_name, node_info in graph.node_list.items():
        if node_info.type == "Merge" or node_info.type == "Route":
            custom_list[node_name] = node_info

        for inp in node_info.inputs:
            input2node[inp].append(node_name)

        for out in node_info.outputs:
            output2node[out].append(node_name)

    return input2node, output2node, custom_list

def find_equivalent_sets(node_dict, g: Graph, flag: bool = True):
    node_to_set = {}
    equivalent_sets = defaultdict(list)
    set_keys = defaultdict(set)
    
    set_index = 0
    equal = g.can_batch
    for key, node_list in node_dict.items():
        for node in node_list:
            if (node, key) not in node_to_set:
                found_equivalent = False
                for (existing_node, existing_key), existing_set in node_to_set.items():
                    try:
                        if equal(g.node_list[node], g.node_list[existing_node], True) and key not in set_keys[existing_set]:
                            equivalent_sets[existing_set].append(node)
                            node_to_set[(node, key)] = existing_set
                            set_keys[existing_set].add(key)
                            found_equivalent = True
                            break
                    except:
                        continue
                
                if not found_equivalent:
                    node_to_set[(node, key)] = set_index
                    # equivalent_sets[set_index].append((node, key))
                    equivalent_sets[set_index].append(node)
                    set_keys[set_index].add(key)
                    set_index += 1

    return list(equivalent_sets.values())

def eliminate_op(graph: Graph):
    def get_route(ls: list[str]) -> list[str]:
        lset = set(ls)
        return [item for item in lset if item]
    input2node, output2node, custom_list = get_map(graph)
    """
    消解规则：考虑Route-Merge模式
        1. 我们认为如果Route节点的输出 包含于 Merge节点的输入，那么此时我们可以将该route删除；
        2. 如果Merge节点只有一个输入，那么消去该节点。

    此时，route的输出实际上可以作为多个节点的输入，这里使用我们的规则可以处理。
    We only consider from the view of Merge node.
    """
    for node_name, node_info in custom_list.items():
        if node_info.type == "Merge":
            route_name = [''] * len(node_info.inputs)
            for idx, inp in enumerate(node_info.inputs):
                if inp not in output2node:
                    continue
                cur_node = output2node[inp][0]
                if graph.node_list[cur_node].type == "Route":
                    route_name[idx] = cur_node
            route_name = get_route(route_name)
            assert '' not in route_name
            for name in route_name:
                route_node = graph.node_list[name]
                route_output = set(route_node.outputs)
                merge_input = set(node_info.inputs)
                if route_output == merge_input and route_node.axis == node_info.axis:
                    route_inp = route_node.inputs[0]
                    if True:
                        new_node = Node(node_name + '_id', "Identity", [route_inp], [node_info.outputs[0]], [], [],input_index=[0])
                        graph.add_node(new_node, False)
                        del graph.node_list[node_info.name]


def remove_identity(graph: Graph):
    remove_key = []
    for node_name, node_info in graph.node_list.items():
        if node_info.type == "Identity":
            if len(node_info.inputs) != 1 or len(node_info.outputs) != 1:
                continue
            node_input = node_info.inputs[0]
            node_output = node_info.outputs[0]
            remove_key.append(node_name)
            try:
                node_id = graph.output.index(node_output)
                graph.output[node_id] = node_input
            except:
                for nn_name, nnode_info in graph.node_list.items():
                    if nn_name != node_name:
                        for idx, inp in enumerate(nnode_info.inputs):
                            if inp == node_output:
                                nnode_info._inputs[idx][0] = node_input
                                # nnode_info.inputs[idx] = node_input

    for key in remove_key:
        del graph.node_list[key]


def fuse_other(graph: Graph, model_num: int = 2) -> bool:
    """
    可以泛化到多个输出的情况

    讨论Route节点的每个输出连接到哪些节点
    例如：
    out1 -> [A1, C1] 
    out2 -> [B2, C2]
    out3 -> [A3, B3]

    我们需要找到可以batch的节点
    [A1, A3]
    [B2, B3]
    [C1, C2]

    对于这些节点我们需要再次进行fusion 操作：
    1. 将这些节点fuse 为一个节点，其中如果该节点不止一个输入，那么我们对他们的每个index的输入都加一个Merge Op
    2. 在这个fuse的节点后添加一个Route Op。


    事实上我们只需要 Route-Merge 消解 的 情况，其他的情况仍是算子融合。

    对Merge节点的每个输入，注意这里的输入只能作为一个节点的输出【重要】
    我们有：
    in1 -> [A1]
    in2 -> [A2]
    in3 -> [B3]
    得到batch 的节点 [A1, A2], 这里直接将这两个节点融合，并在上一层添加Merge节点；这里其实最好还是添加Route节点。

    """
    has_change = False
    input2node, output2node , custom_list = get_map(graph)
    custom_list_info = list(custom_list.values())
    for node_info in custom_list_info:
        if node_info.type == "Route":
            node_dict = {}
            for out in node_info.outputs:
                if out in input2node:
                    node_dict[out] = input2node[out]
            equivalent_sets = find_equivalent_sets(node_dict, graph)

            for eqset in equivalent_sets:
                if len(eqset) > 1:
                # if len(eqset) == model_num:
                    has_change = True
                    for eqs in eqset:
                        if eqs not in graph.node_list:
                            has_change = False
                            break
                    if not has_change:
                        continue
                    op = graph.node_list[eqset[0]]
                    assert op.type != 'Route' and op.type != 'Merge' , "Cannot be Route or Merge"
                    op_input = [[inp] for inp in op.inputs]
                    op_output = [[out] for out in op.outputs]

                    for eqs in eqset[1:]:
                        cur_node = graph.node_list[eqs]
                        for idx, inp in enumerate(cur_node.inputs):
                            op_input[idx].append(inp)
                        for idx, out in enumerate(cur_node.outputs):
                            op_output[idx].append(out)
                        del graph.node_list[eqs]

                    for idx, inp in enumerate(op_input):
                        assert len(inp) > 1
                        new_name = f"{op.name}_{idx}_merge"
                        axis = graph.edge_list[inp[0]].batch_dim
                        # assert axis != -1, "Non-valid axis."
                        merge_op = Node(new_name, 'Merge', inp, [new_name + '_m1o2c3',], [], [],axis=axis)
                        op._inputs[idx] = [new_name + '_m1o2c3']

                        graph.add_node(merge_op,False)
                        for idxx, iinp in enumerate(inp):
                            input2node[iinp].remove(eqset[idxx])
                            input2node[iinp].append(new_name)

                    for idx, out in enumerate(op_output):
                        assert len(out) > 1
                        new_name = f"{op.name}_{idx}_route"
                        axis = graph.edge_list[out[0]].batch_dim
                        # assert axis != -1, "Non-valid axis."
                        route_op = Node(new_name, 'Route', [new_name + '_m2o1c3'], out, [], [], axis=axis)
                        route_op.gather_list = get_group(out)
                        op._outputs[idx] = [new_name + '_m2o1c3']
                        graph.add_node(route_op,False)
                        custom_list_info.append(route_op)
                        for idxx, oout in enumerate(out):
                            output2node[oout].remove(eqset[idxx])
                            output2node[oout].append(new_name)

        elif node_info.type == "Merge":
            # 只需要考虑fuse节点的输出不能作为多个节点的输入
            node_dict = {}
            for inp in node_info.inputs:
                if inp in output2node:
                    node_dict[inp] = output2node[inp]
            equivalent_sets = find_equivalent_sets(node_dict, graph, False)

            for eqset in equivalent_sets:
                if len(eqset) > 1:
                # if len(eqset) == model_num:
                    has_change = True
                    op = graph.node_list[eqset[0]]
                    op_input = [[inp] for inp in op.inputs]
                    op_output = [[out] for out in op.outputs]

                    for eqs in eqset[1:]:
                        cur_node = graph.node_list[eqs]
                        for idx, inp in enumerate(cur_node.inputs):
                            op_input[idx].append(inp)
                        for idx, out in enumerate(cur_node.outputs):
                            op_output[idx].append(out)

                        del graph.node_list[eqs]
                    

                    for idx, inp in enumerate(op_input):
                        assert len(inp) > 1
                        new_name = f"{op.name}_{idx}_merge" #op_input[idx][0] + '_merge'
                        axis = graph.edge_list[inp[0]].batch_dim
                        # assert axis != -1, "Non-valid axis."
                        merge_op = Node(new_name, 'Merge', inp, [new_name + '_m1o2c3',], [], [],axis=axis)
                        op._inputs[idx] = [new_name + '_m1o2c3']
                        graph.add_node(merge_op, False)
                        custom_list_info.append(merge_op)
                        for idxx, iinp in enumerate(inp):
                            input2node[iinp].remove(eqset[idxx])
                            input2node[iinp].append(new_name)

                    for idx, out in enumerate(op_output):
                        assert len(out) > 1
                        new_name = f"{op.name}_{idx}_route"
                        axis = graph.edge_list[out[0]].batch_dim
                        # assert axis != -1, "Non-valid axis."
                        route_op = Node(new_name, 'Route', [new_name + '_m2o1c3'], out, [], [], axis=axis)
                        route_op.gather_list = get_group(out)
                        op._outputs[idx] = [new_name + '_m2o1c3']
                        graph.add_node(route_op, False)    
                        for idxx, oout in enumerate(out):
                            output2node[oout].remove(eqset[idxx])
                            output2node[oout].append(new_name)         


    return has_change

def clean_unused_node(graph: Graph):
    has_change = True
    while has_change:
        all_input = set()
        for node in graph.node_list.values():
            all_input.update(node.inputs)
        assert isinstance(graph.output, list), "Output should be a list"
        all_input.update(graph.output)
        node_list = []
        for name, node in graph.node_list.items():
            if all_input.isdisjoint(node.outputs):
                node_list.append(name)
        
        remove_constants = []
        has_change = len(node_list) > 0 or len(remove_constants) > 0
        for node in node_list:
            del graph.node_list[node]

        for cons in remove_constants:
            del graph.constants[cons]

def combine(graph: Graph):
    has_change = True
    index = 0
    while has_change:
    # if True:
        index += 1
        # fuse_other(graph)
        # eliminate_op(graph)
        # remove_identity(graph)
        # clean_unused_node(graph)
        # has_change = False
        
        has_change = fuse_other(graph)
        if not has_change:
            clean_unused_node(graph)
            break
        print(f'{index} try.')
        eliminate_op(graph)
        remove_identity(graph)
        clean_unused_node(graph)