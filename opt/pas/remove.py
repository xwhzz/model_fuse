from opt.graph import *
from collections import defaultdict
from collections import deque

def get_map(graph: Graph):
    input2node = {}
    output2node = {} 
    custom_list = {} 

    for node_name, node_info in graph.node_list.items():
        if node_info.Type == "Merge" or node_info.Type == "Route":
            custom_list[node_name] = node_info

        for inp in node_info.Input:

            if inp not in input2node:
                input2node[inp] = []
            input2node[inp].append(node_name)
        for out in node_info.Output:

            if out not in output2node:
                output2node[out] = []
            output2node[out].append(node_name)

    return input2node, output2node, custom_list

def find_equivalent_sets(node_dict, g: Graph, flag: bool = True):

    node_to_set = {}

    equivalent_sets = defaultdict(list)

    set_keys = defaultdict(set)
    
    set_index = 0
    equal = g.weight_is_equal
    for key, node_list in node_dict.items():
        for node in node_list:
            if (node, key) not in node_to_set:
                found_equivalent = False
                for (existing_node, existing_key), existing_set in node_to_set.items():
                    if flag:
                        ekey_index = g.node_list[existing_node].Input.index(existing_key)
                        key_index = g.node_list[node].Input.index(key)
                    else:
                        ekey_index = g.node_list[existing_node].Output.index(existing_key)
                        key_index = g.node_list[node].Output.index(key)
                    if equal(g.node_list[node], g.node_list[existing_node], key_index, ekey_index) and key not in set_keys[existing_set]:
                        equivalent_sets[existing_set].append(node)
                        node_to_set[(node, key)] = existing_set
                        set_keys[existing_set].add(key)
                        found_equivalent = True
                        break
                
                if not found_equivalent:
                    node_to_set[(node, key)] = set_index
                    # equivalent_sets[set_index].append((node, key))
                    equivalent_sets[set_index].append(node)
                    set_keys[set_index].add(key)
                    set_index += 1

    return list(equivalent_sets.values())

def find_chains(tuples: list[tuple[str,str]]) -> list[list[str]]:
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for src, dst in tuples:
        graph[src].append(dst)
        in_degree[dst] += 1
        if src not in in_degree:
            in_degree[src] = 0
    
    sources = [node for node, degree in in_degree.items() if degree == 0]
    def dfs(node, path, paths):
        if not graph[node]: 
            paths.append(path)
            return
        
        for neighbor in graph[node]:
            dfs(neighbor, path + [neighbor], paths)
    
    all_paths = []
    for source in sources:
        paths = []
        dfs(source, [source], paths)
        all_paths.extend(paths)
    
    return all_paths


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
    queue_list = []
    for node_name, node_info in custom_list.items():
        if node_info.Type == "Merge":
            route_name = [''] * len(node_info.Input)
            for idx, inp in enumerate(node_info.Input):
                cur_node = output2node[inp][0]
                if graph.node_list[cur_node].Type == "Route":
                    route_name[idx] = cur_node
            route_name = get_route(route_name)
            for name in route_name:
                route_node = graph.node_list[name]
                route_output = set(route_node.Output)
                merge_input = set(node_info.Input)
                merge_input_ls = list(merge_input)
                if route_output.issubset(merge_input):
                    for idx, out in enumerate(route_node.Output):
                        merge_input_ls.remove(out) 
                    route_inp = route_node.Input[0]
                    merge_input_ls.append(route_inp)

                    if len(merge_input_ls) == 1:
                        # Process route information.
                        # 可能不存在下一个Route节点，这里的例子是最后的输出总需要一个Merge
                        if node_info.Output[1] in input2node:
                            for nnode in input2node[node_info.Output[1]]:
                                if graph.node_list[nnode].Type == "Route":
                                    queue_list.append((name, nnode))

                        new_node = NodeInfo("Identity", [route_inp] , [node_info.Output[0]], [], None)
                        graph.add_node(new_node, node_name + '_id')

                    node_info.Input = merge_input_ls
    queue_list = find_chains(queue_list)
    for queue in queue_list:
        for q in queue[1:]:
            graph.node_list[q].Input[1] = graph.node_list[queue[0]].Input[1]


def remove_op(graph: Graph):
    has_change = False
    try_times = 0
    input2node, output2node, custom_list = get_map(graph)
    remove_node = set()
    while has_change or try_times == 0:
        try_times += 1
        has_change = False
        custom_list = {k: v for k, v in custom_list.items() if k not in remove_node}
        remove_node = set()
        for node_name, node_info in custom_list.items():
            if node_name in remove_node:
                continue
            if node_info.Type == "Merge":
                op_type = []
                other_list = []
                good_node = 0
                for inp in node_info.Input:
                    try:
                        if len(input2node[inp]) == 1:
                            before_merge = graph.node_list[output2node[inp][0]]
                            ## TODO: 考虑有weight的算子
                            if before_merge.Can_batch: #and (not before_merge.has_weight(False)):
                                good_node += 1
                                other_list.append(before_merge.Other)
                            op_type.append(before_merge.Type)
                    except:
                        break

                if len(node_info.Input) == len(op_type):
                    ## 注意我们在此处考虑Merge 算子的上一层是Route的情况，我们消解算子只需要考虑这种情况
                    op_typ = op_type[0]
                    other = other_list[0] if other_list else None
                    if all((type == op_typ for type in op_type[1:])):
                        if op_typ == "Route":
                            route_name = []
                            for inp in node_info.Input:
                                if len(input2node[inp]) == 1:
                                    # print(f"input {inp}: ",input2node[inp])
                                    route_name.append(output2node[inp][0])
                            try:
                                route_name_ = route_name[0]
                            except:
                                continue

                            if len(node_info.Input) == len(route_name) and all((route_name_ == nn for nn in route_name[1:])):
                                """
                                我们直接使用Identity 算子 
                                输入： route的输入[0]
                                输出： merge的输出[0]

                                除此之外我们需要考虑：
                                route的输入[1] 应该找到merge的输出[1]对应的route节点，将其变换
                                """
                                route_op = graph.node_list[route_name_]                            
                                has_change = True
                                new_node_name = route_name_ + "_Identity"

                                input_index = input2node[route_op.Input[0]].index(route_name_)
                                input2node[route_op.Input[0]][input_index] = new_node_name

                                output_index = output2node[node_info.Output[0]].index(node_name)
                                output2node[node_info.Output[0]][output_index] = new_node_name

                                for inp in node_info.Input:
                                    del input2node[inp]
                                    del output2node[inp]

                                if node_info.Output[1] in input2node:
                                    next_route = graph.node_list[input2node[node_info.Output[1]][0]]
                                    next_route.Input[1] = route_op.Input[1] 
                                new_node = NodeInfo("Identity", [route_op.Input[0]] , [node_info.Output[0]], [], None)
                                del graph.node_list[route_name_]
                                del graph.node_list[node_name]
                                graph.add_node(new_node, new_node_name)

                                remove_node.add(route_name_)
                                remove_node.add(node_name)

                        elif good_node == len(node_info.Input) and all((other == oth for oth in other_list[1:])):
                            node_1 = graph.node_list[output2node[node_info.Input[0]][0]]
                            ## 判断weight 以及 attr
                            same_weight = True
                            for node_inp in node_info.Input[1:]:
                                node_2 = graph.node_list[output2node[node_inp][0]]
                                if not graph.weight_is_equal(node_1, node_2):
                                    same_weight = False
                                    break
                            if not same_weight:
                                continue
                            has_change = True
                            node_info_input = []
                            new_node_name = None
                            """
                            如果所有节点具有相同类型
                                删除所有节点，并配置一个新的节点，并将其放在Merge算子的下一层

                                注意这里的参数关系

                                Merge节点现在的输入 应该是 【之前所有节点的输入】 这些参数的input变成merge， output不变
                                Merge节点现在的第一个输出 是一个 新值，注意此时第二个
                                新节点的输入是 这个新值
                                新节点的输出是 Merge节点 之前的输出 -> Merge节点的输出 参数，output变化，input不变
                            """
                            for idx, inp in enumerate(node_info.Input):
                                if idx == 0:
                                    new_node_name = output2node[inp][0] 
                                cur_node = graph.node_list[output2node[inp][0]]
                                node_info_input.append(cur_node.Input[0])

                                input_index = input2node[cur_node.Input[0]].index(output2node[inp][0])
                                input2node[cur_node.Input[0]][input_index] = node_name

                                del graph.node_list[output2node[inp][0]]

                            node_info.Input = node_info_input
                            output_ = node_info.Output[0]
                            new_name = output_ + '_m1o2c3'
                            node_info.Output[0] = new_name

                            input2node[new_name] = [new_node_name]
                            output2node[new_name] = [node_name]
                            cur_index = output2node[output_].index(node_name)
                            output2node[output_][cur_index] = new_node_name
                            new_node = NodeInfo(op_typ, [new_name] , [output_], node_1.Parameters, other, InputIndex=node_1.InputIndex)
                            graph.add_node(new_node, new_node_name)
            elif node_info.Type == "Route":
                op_type = []
                other_list = []
                for out in node_info.Output:
                    try:
                        ## TODO: 我们可以讨论route的输出可以作为多个节点输入的情况
                        """
                        我们只讨论输出为单个的算子

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

                        消解规则：
                            1. 如果为一一映射，即Route的所有输出为Merge的所有输入，那么添加Identity节点。
                            2. 如果Merge的所有输入包含Route的所有输出，那么此时 去除 Route 节点；
                            3. 如果Route的所有输出包含Merge的所有输入，不考虑这种情况。【暂时不考虑】
                        
                            
                        我们这么操作，每次迭代先添加后消解
                        """
                        if True:
                        # if len(input2node[out]) == 1:
                            after_route = graph.node_list[input2node[out][0]]
                            if after_route.Type != "Merge" and after_route.Can_batch: #and (not after_route.has_weight(False)):
                                op_type.append(after_route.Type)
                                other_list.append(after_route.Other)
                    except:
                        break

                if len(node_info.Output) == len(op_type):
                    op_typ = op_type[0]
                    other = other_list[0]
                    if all((type == op_typ for type in op_type[1:])) and all((other == oth for oth in other_list[1:])):
                        node_1 = graph.node_list[input2node[node_info.Output[0]][0]]
                        same_weight = True
                        for node_inp in node_info.Output[1:]:
                            node_2 = graph.node_list[input2node[node_inp][0]]
                            if not graph.weight_is_equal(node_1, node_2):
                                same_weight = False
                                break
                        if not same_weight:
                            continue
                        has_change = True
                        node_info_output = []
                        new_node_name = None
                        """
                        如果所有节点具有相同类型
                            删除所有节点，并配置一个新的节点，并将其放在Route算子的上一层

                            注意这里的参数关系

                            Route 节点现在的输出 应该是 【之前所有节点的输出】 这些参数的output变成route， input不变
                            Merge节点现在的第一个输出 是一个 新值，注意此时第二个
                            新节点的输入是 这个新值
                            新节点的输出是 Merge节点 之前的输出 -> Merge节点的输出 参数，output变化，input不变
                        """
                        for idx, out in enumerate(node_info.Output):
                            if idx == 0:
                                new_node_name = input2node[out][0] 
                            cur_node = graph.node_list[input2node[out][0]]
                            node_info_output.append(cur_node.Output[0])

                            output_index = output2node[cur_node.Output[0]].index(input2node[out][0])
                            output2node[cur_node.Output[0]][output_index] = node_name 

                            del graph.node_list[input2node[out][0]]

                        node_info.Output = node_info_output
                        input_ = node_info.Input[0]
                        new_name = input_ + '_m1o2c3'
                        node_info.Input[0] = new_name

                        input2node[new_name] = [node_name]
                        output2node[new_name] = [new_node_name]

                        cur_index = input2node[input_].index(node_name)
                        input2node[input_][cur_index] = new_node_name
                        para_list = node_1.Parameters
                        index_list = node_1.InputIndex
                        new_node = NodeInfo(op_typ, [input_] , [new_name], para_list, other, InputIndex=index_list)
                        graph.add_node(new_node, new_node_name)

    remove_identity(graph)

def remove_identity(graph: Graph):
    remove_key = []
    for node_name, node_info in graph.node_list.items():
        if node_info.Type == "Identity":
            node_input = node_info.Input[0]
            node_output = node_info.Output[0]
            remove_key.append(node_name)
            try:
                node_id = graph.output.index(node_output)
                graph.output[node_id] = node_input
            except:
                for nn_name, nnode_info in graph.node_list.items():
                    if nn_name != node_name:
                        for idx, inp in enumerate(nnode_info.Input):
                            if inp == node_output:
                                nnode_info.Input[idx] = node_input

    for key in remove_key:
        del graph.node_list[key]


def fuse_other(graph: Graph) -> bool:
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
        if node_info.Type == "Route":
            node_dict = {}
            for out in node_info.Output:
                if out in input2node:
                    node_dict[out] = input2node[out]
            # print(node_dict)
            equivalent_sets = find_equivalent_sets(node_dict, graph)
            # print(equivalent_sets)
            for eqset in equivalent_sets:
                if len(eqset) > 1:
                    has_change = True
                    op = graph.node_list[eqset[0]]
                    op_input = [[inp] for inp in op.Input]
                    op_output = [[out] for out in op.Output]

                    for eqs in eqset[1:]:
                        cur_node = graph.node_list[eqs]
                        for idx, inp in enumerate(cur_node.Input):
                            op_input[idx].append(inp)
                        for idx, out in enumerate(cur_node.Output):
                            op_output[idx].append(out)

                        del graph.node_list[eqs]

                    for idx, inp in enumerate(op_input):
                        assert len(inp) > 1
                        merge_op = NodeInfo('Merge', inp, [op_input[idx][0] + '_m1o2c3', op_input[idx][0] + '_a1d2d3'], [], None)
                        op.Input[idx] = op_input[idx][0] + '_m1o2c3'
                        new_name = op_input[idx][0] + '_merge'
                        graph.add_node(merge_op, new_name)
                        # input2node[inp] = new_name
                        for idxx, iinp in enumerate(inp):
                            # print(iinp)
                            # # index = input2node[iinp].index(iinp)
                            # input2node[input2node[iinp].index()] = new_name
                            # input2node[iinp].append(new_name)
                            # for eqs in eqset:
                            input2node[iinp].remove(eqset[idxx])
                            input2node[iinp].append(new_name)


                    for idx, out in enumerate(op_output):
                        assert len(out) > 1
                        route_op = NodeInfo('Route', [op_output[idx][0] + '_m2o1c3', op_input[0][0] + '_a1d2d3'], out, [], None)
                        op.Output[idx] = op_output[idx][0] + '_m2o1c3'
                        new_name = op_output[idx][0] + '_route'
                        graph.add_node(route_op, new_name)
                        ## 这里我们打算将route节点加入到需要测试的节点中。
                        custom_list_info.append(route_op)
                        for idxx, oout in enumerate(out):
                            output2node[oout].remove(eqset[idxx])
                            output2node[oout].append(new_name)

        elif node_info.Type == "Merge":
            # 只需要考虑fuse节点的输出不能作为多个节点的输入
            node_dict = {}
            for inp in node_info.Input:
                if inp in output2node:
                    node_dict[inp] = output2node[inp]
            equivalent_sets = find_equivalent_sets(node_dict, graph, False)

            for eqset in equivalent_sets:
                if len(eqset) > 1:
                    has_change = True
                    op = graph.node_list[eqset[0]]
                    op_input = [[inp] for inp in op.Input]
                    op_output = [[out] for out in op.Output]

                    for eqs in eqset[1:]:
                        cur_node = graph.node_list[eqs]
                        for idx, inp in enumerate(cur_node.Input):
                            op_input[idx].append(inp)
                        for idx, out in enumerate(cur_node.Output):
                            op_output[idx].append(out)

                        del graph.node_list[eqs]
                    
                    # # 需要fuse节点的输出不能作为多个节点的输入
                    # for out_ in op_output:
                    #     for out in out_:
                    #         if len(input2node[out]) > 1:

                                
                    # 开始加节点

                    for idx, inp in enumerate(op_input):
                        assert len(inp) > 1
                        merge_op = NodeInfo('Merge', inp, [op_input[idx][0] + '_m1o2c3', op_input[idx][0] + '_a1d2d3'], [], None)
                        op.Input[idx] = op_input[idx][0] + '_m1o2c3'
                        new_name = op_input[idx][0] + '_merge'
                        graph.add_node(merge_op, new_name)
                        custom_list_info.append(merge_op)
                        for idxx, iinp in enumerate(inp):
                            input2node[iinp].remove(eqset[idxx])
                            input2node[iinp].append(new_name)


                    for idx, out in enumerate(op_output):
                        assert len(out) > 1
                        route_op = NodeInfo('Route', [op_output[idx][0] + '_m2o1c3', op_input[0][0] + '_a1d2d3'], out, [], None)
                        op.Output[idx] = op_output[idx][0] + '_m2o1c3'
                        new_name = op_output[idx][0] + '_route'
                        graph.add_node(route_op, new_name)    
                        for idxx, oout in enumerate(out):
                            output2node[oout].remove(eqset[idxx])
                            output2node[oout].append(new_name)         

    return has_change

def clean_unused_node(graph: Graph):
    has_change = True
    while has_change:
        has_change = False
        all_input = set()
        for node in graph.node_list.values():
            all_input.update(node.Input)
        assert isinstance(graph.output, list), "Output should be a list"
        all_input.update(graph.output)
        node_list = []
        for name, node in graph.node_list.items():
            if all_input.isdisjoint(node.Output):
                node_list.append(name)
        has_change = len(node_list) > 0
        for node in node_list:
            del graph.node_list[node]

def combine(graph: Graph):
    has_change = True
    index = 0
    while has_change:
        index += 1
        has_change = False
        has_change = fuse_other(graph)
        if not has_change:
            clean_unused_node(graph)
            break
        print(f'{index} try.')
        eliminate_op(graph)
        remove_identity(graph)
        clean_unused_node(graph)