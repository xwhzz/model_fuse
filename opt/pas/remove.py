from opt.graph import *

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
                            has_weight = node_1.has_weight(False)
                            if has_weight:
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
                            if has_weight:
                                new_node = NodeInfo(op_typ, [new_name] , [output_], node_1.Parameters, other, InputIndex=node_1.InputIndex)
                            else:
                                new_node = NodeInfo(op_typ, [new_name] , [output_], [], other, InputIndex=[0])
                            graph.add_node(new_node, new_node_name)
            elif node_info.Type == "Route":
                op_type = []
                other_list = []
                for out in node_info.Output:
                    try:
                        if len(input2node[out]) == 1:
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
                        has_weight = node_1.has_weight(False)
                        if has_weight:
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
                        # if has_weight:
                        if has_weight:
                            new_node = NodeInfo(op_typ, [input_] , [new_name], para_list, other, InputIndex=index_list)
                        else:
                            new_node = NodeInfo(op_typ, [input_] , [new_name], [], other, InputIndex=[0])
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