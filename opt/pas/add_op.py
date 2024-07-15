from opt.graph import *
from copy import deepcopy


def add_op(graph: Graph):
    node_list = deepcopy(graph.node_list)

    for node_name, node_info in node_list.items():
        if node_info.has_weight() and len(node_info.Input) > 1:
            assert len(node_info.Input) == len(node_info.Output)
            merge_op = NodeInfo('Merge', node_info.Input, [node_name + '_m1o2c3', node_name + '_a1d2d3'], [], None)
            route_op = NodeInfo('Route', [node_name + '_m2o1c3', node_name + '_a1d2d3'], node_info.Output, [], None)
            op = graph.node_list[node_name]
            op.Input = [node_name+'_m1o2c3']
            op.Output = [node_name+'_m2o1c3']
            graph.add_node(merge_op, node_name + '_merge')
            graph.add_node(route_op, node_name + '_route')
    
    if len(graph.input) > 1:
        route_op = NodeInfo('Route', ['Input', 'Info_1'],graph.input, [], None)
        merge_op = NodeInfo('Merge', graph.output, ['Output', 'Info_add'], [], None)

        graph.add_node(route_op, 'route')
        graph.add_node(merge_op, 'merge')

        graph.input = ['Input', "Info_1"]
        graph.output = ['Output']

