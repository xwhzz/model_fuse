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
            # 不考虑Route的输入
            if node_info.Type == "Route":
                break
            if inp not in input2node:
                input2node[inp] = []
            input2node[inp].append(node_name)
        for out in node_info.Output:
            # 不考虑Merge的输出
            if node_info.Type == "Merge":
                break
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
        # print(f"{try_times}th try!")
        has_change = False
        for node_name, node_info in custom_list.items():
            if node_info.Type == "Merge":
                op_type = []
                for inp in node_info.Input:
                    if inp in output2node:
                        # 得到merge的上一层节点
                        before_merge = graph.node_list[output2node[inp]]
                        # 这里如果上一层是route，我们先忽略；并且我们假设上一层节点的输出只能作为一个节点的输入
                        if before_merge.Type != "Route" and before_merge.Output[0] in input2node:
                            op_type.append(before_merge.Type)

                if len(node_info.Input) == len(op_type):
                    op_typ = op_type[0]
                    if all((type == op_typ for type in op_type[1:])):
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

            elif node_info.Type == "Route":
                ...