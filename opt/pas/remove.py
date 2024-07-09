from opt.graph import *

def get_map(graph: Graph):
    input2node = {} # 数据作为某个节点的输入
    output2node = {} # 数据作为某个节点的输出
    custom_list = {} # 我们自定义的算子列表

    for node_name, node_info in graph.node_list.items():
        if node_info.Type == "Merge" or node_info.Type == "Route":
            custom_list[node_name] = node_info
        elif len(node_info.Input) > 1 or len(node_info.Output) > 1:
            continue
        for inp in node_info.Input:
            # x 不考虑Route的输入
            # if node_info.Type == "Route":
            #     break
            if inp not in input2node:
                input2node[inp] = []
            input2node[inp].append(node_name)
        for out in node_info.Output:
            # x 不考虑Merge的输出
            # if node_info.Type == "Merge":
            #     break
            if out not in output2node:
                output2node[out] = []
            output2node[out].append(node_name)
    # 如果某个数据作为多个节点的输入，我们直接清除
    input2node = {k: v[0] for k, v in input2node.items() if len(v) == 1}
    output2node = {k: v[0] for k, v in output2node.items() if len(v) == 1}

    return input2node, output2node, custom_list
    

def remove_op(graph: Graph):
    has_change = False
    try_times = 0

    input2node, output2node, custom_list = get_map(graph)

    while has_change or try_times == 0:
        try_times += 1
        has_change = False
        for node_name, node_info in custom_list.items():
            if node_info.Type == "Merge":
                op_type = []
                for inp in node_info.Input:
                    if inp in output2node:
                        # 得到merge的上一层节点
                        before_merge = graph.node_list[output2node[inp]]
                        # 我们假设上一层节点的输出只能作为一个节点的输入
                        if before_merge.Output[0] in input2node:
                            op_type.append(before_merge.Type)
                            
                if len(node_info.Input) == len(op_type):
                    ## 注意我们在此处考虑Merge 算子的上一层是Route的情况，我们消解算子只需要考虑这种情况
                    op_typ = op_type[0]
                    if all((type == op_typ for type in op_type[1:])):
                        if op_typ == "Route":
                            route_name = []
                            for inp in node_info.Input:
                                route_name.append(output2node[inp])
                            route_name_ = route_name[0]
                            ## 如果所有节点的名字相同
                            if all((route_name_ == nn for nn in route_name[1:])):
                                """
                                我们直接使用Identity 算子 
                                输入： route的输入[0]
                                输出： merge的输出[0]

                                除此之外我们需要考虑：
                                route的输入[1] 应该找到merge的输出[1]对应的route节点，将其变换
                                """

                                route_op = graph.node_list[route_name_]
                                flag = False
                                for inp in node_info.Input:
                                    if inp not in input2node:
                                        flag = True
                                        break
                                if flag:
                                    continue
                                has_change = True
                                new_node_name = route_name_ + "_Identity"

                                input2node[route_op.Input[0]] = new_node_name
                                output2node[node_info.Output[0]] = new_node_name

                                for inp in node_info.Input:
                                    del input2node[inp]
                                    del output2node[inp]

                                if node_info.Output[1] in input2node:
                                    next_route = graph.node_list[input2node[node_info.Output[1]]]
                                    next_route.Input[1] = node_info.Output[1]
                                new_node = NodeInfo("Identity", [route_op.Input[0]] , [node_info.Output[0]], [], None)
                                del graph.node_list[route_name_]
                                del graph.node_list[node_name]
                                graph.add_node(new_node, new_node_name)

                        else:
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
                                    new_node_name = output2node[inp] 
                                cur_node = graph.node_list[output2node[inp]]
                                node_info_input.append(cur_node.Input[0])

                                input2node[cur_node.Input[0]] = node_name

                                del graph.node_list[output2node[inp]]

                            node_info.Input = node_info_input
                            output_ = node_info.Output[0]
                            new_name = output_ + '_sum'
                            node_info.Output[0] = new_name

                            input2node[new_name] = new_node_name
                            output2node[new_name] = node_info

                            output2node[output_] = new_node_name
                            new_node = NodeInfo(op_typ, [new_name] , [output_], [], None)
                            graph.add_node(new_node, new_node_name)
            # elif False:
            elif node_info.Type == "Route":
                op_type = []
                for out in node_info.Output:
                    if out in input2node:
                        # 得到Route 的下一层节点
                        after_route = graph.node_list[input2node[out]]
                        # 这里如果下一层是merge，我们先忽略；并且我们假设下一层节点的输入只能作为一个节点的输出
                        if after_route.Type != "Merge" and after_route.Input[0] in output2node:
                            op_type.append(after_route.Type)

                if len(node_info.Output) == len(op_type):
                    op_typ = op_type[0]
                    if all((type == op_typ for type in op_type[1:])):
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
                                new_node_name = input2node[out] 
                            cur_node = graph.node_list[input2node[out]]
                            node_info_output.append(cur_node.Output[0])

                            output2node[cur_node.Output[0]] = node_name

                            del graph.node_list[input2node[out]]

                        node_info.Output = node_info_output
                        input_ = node_info.Input[0]
                        new_name = input_ + '_sum'
                        node_info.Input[0] = new_name

                        input2node[new_name] = node_info
                        output2node[new_name] = new_node_name
                        ## Route节点的Input 现在作为 新节点的输入
                        input2node[input_] = new_node_name
                        new_node = NodeInfo(op_typ, [input_] , [new_name], [], None)
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