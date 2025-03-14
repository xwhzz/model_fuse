from ..graph import *
from ..utils import *
from copy import deepcopy

# at least make the inputs and outputs of graph 1 and graph 2 the same,
# specify the index of the input and output with no batch size
def add_op(graph: Graph, no_batch_input: list[int] | None = None, no_batch_output: list[int] | None = None):
    node_list = deepcopy(graph.node_list)
    res = 0
    for node_name, node_info in node_list.items():
        # if len(node_info.fuse) > 1:
        if node_info.fuse:
            res += 1
            for idx, inp in enumerate(node_info._inputs):
                assert len(inp) > 1, "Input should have more than one element."
                axis = graph.edge_list[inp[0]].batch_dim
                # assert axis != -1, "Batch dimension should be specified."
                new_name = f"{node_name}_{idx}_merge"
                merge_op = Node(new_name,'Merge', inp, [new_name + '_m1o2c3'], [], [], axis=axis)
                graph.node_list[node_name]._inputs[idx] = [new_name + '_m1o2c3']
                graph.add_node(merge_op, False)
                # print(node_info._inputs[idx])
            for idx, out in enumerate(node_info._outputs):
                assert len(out) > 1, "Output should have more than one element."
                axis = graph.edge_list[out[0]].batch_dim
                # assert axis != -1, "Batch dimension should be specified."
                new_name = f"{node_name}_{idx}_route"
                split_op = Node(new_name, 'Route', [new_name + '_m1o2c3'], out, [], [], axis=axis)
                split_op.gather_list = get_group(out)
                graph.node_list[node_name]._outputs[idx] = [new_name + '_m1o2c3']
                graph.add_node(split_op, False)

    input_list = []
    output_list = []

    # process the input
    for idx, inp in enumerate(graph.input):
        if len(inp) == 1 or (no_batch_input is not None and idx in no_batch_input):
            input_list.extend(inp)
            continue
        route_name = f'route_{idx}'
        inp_name = f'input_{idx}'
        axis = graph.edge_list[inp[0]].batch_dim
        route_op = Node(route_name,'Route', [inp_name,], inp, [], [], axis=axis)
        route_op.gather_list = get_group(inp)
        
        graph.add_node(route_op, False)
        input_list.append(inp_name)
    input_list.append('info')
    graph.input = input_list

    # process the output
    for idx, out in enumerate(graph.output):
        if len(out) == 1 or (no_batch_output is not None and idx in no_batch_output):
            output_list.extend(out)
            continue
        axis = graph.edge_list[out[0]].batch_dim
        out_name = f'output_{idx}'
        merge_name = f'merge_{idx}'
        merge_op = Node(merge_name,'Merge', out, [f'output_{idx}'], [], [], axis=axis)
        graph.add_node(merge_op, False)
        output_list.append(out_name)

    graph.output = output_list

    return graph

