from opt.graph import *
from copy import deepcopy

# at least make the inputs and outputs of graph 1 and graph 2 the same,
# specify the index of the input and output with no batch size
def add_op(graph: Graph, no_batch_input: list[int] | None = None, no_batch_output: list[int] | None = None):
    node_list = deepcopy(graph.node_list)

    for node_name, node_info in node_list.items():
        if node_info.has_weight() and len(node_info.Input) > 1 and len(node_info.Input) == len(node_info.Output):
            # assert len(node_info.Input) == len(node_info.Output)
            merge_op = NodeInfo('Merge', node_info.Input, [node_name + '_m1o2c3', node_name + '_a1d2d3'], [], None)
            route_op = NodeInfo('Route', [node_name + '_m2o1c3', node_name + '_a1d2d3'], node_info.Output, [], None)
            op = graph.node_list[node_name]
            op.Input = [node_name+'_m1o2c3']
            op.Output = [node_name+'_m2o1c3']
            graph.add_node(merge_op, node_name + '_merge')
            graph.add_node(route_op, node_name + '_route')
    
    input_list = []
    output_list = []

    # process the input
    for idx, inp in enumerate(graph.input):
        if len(inp) == 1 or (no_batch_input is not None and idx in no_batch_input):
            input_list.extend(inp)
            continue
        route_name = f'input_{idx}'
        route_op = NodeInfo('Route', [route_name, 'info'], inp, [], None)
        graph.add_node(route_op, f'route_{idx}')
        input_list.append(route_name)
    input_list.append('info')
    graph.input = input_list

    # process the output
    for idx, out in enumerate(graph.output):
        if len(out) == 1 or (no_batch_output is not None and idx in no_batch_output):
            output_list.extend(out)
            continue
        merge_op = NodeInfo('Merge', out, [f'output_{idx}', f'info_{idx}'], [], None)
        graph.add_node(merge_op, f'merge_{idx}')
        output_list.append(f'output_{idx}')

    graph.output = output_list

    return graph

