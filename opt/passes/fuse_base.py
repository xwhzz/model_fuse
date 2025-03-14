from ..graph import *

def fuse_base(g_1: Graph, g_2: Graph,) -> Graph:
    assert len(g_1.input) == len(g_2.input) and \
        len(g_1.output) == len(g_2.output), \
        "The number of inputs and outputs of the two graphs must be the same"
    fuse_graph = g_1
    fuse_graph.constants.update(g_2.constants)
    fuse_graph.edge_list.update(g_2.edge_list)
    for node_name, node_info in g_2.node_list.items():
        has_same = False
        if len(node_info.parameters) > 0:
            same_ps = set()
            has_same = True
            for idx, para in enumerate(node_info.parameters):
                same = set()
                para_info = g_2.parameter_list[para]
                try:
                    for fuse_para in fuse_graph.parameter_hash[para_info.hash]:
                        fuse_para = fuse_graph.parameter_list[fuse_para]
                        if para_info.value.shape == fuse_para.value.shape and np.allclose(para_info.value, fuse_para.value):
                            name = fuse_para.node
                            cur_node = fuse_graph.node_list[name]
                            if fuse_graph.can_batch(cur_node, node_info):
                                same.add(name)
                except:
                    has_same = False
                    break
                if idx == 0:
                    same_ps = same
                else:
                    same_ps = same_ps & same
                if not same_ps:
                    has_same = False
                    break

            if has_same:
                node_name = same_ps.pop()
                node_fuse = fuse_graph.node_list[node_name]
                node_fuse.add_input_output(node_info.inputs, node_info.outputs)
                node_fuse.fuse = True
                # if not node_fuse.fuse:
                #     node_fuse.fuse.extend(node_info.fuse)

        if not has_same:   
            fuse_graph.node_list[node_name] = node_info
            for para in node_info.parameters:
                para_info = g_2.parameter_list[para]
                fuse_graph.parameter_list[para] = para_info
                fuse_graph.parameter_hash[para_info.hash].append(para)
    
    for idx, inp in enumerate(g_2.input):
        if isinstance(fuse_graph.input[idx], str):
            fuse_graph.input[idx] = [fuse_graph.input[idx]]
        fuse_graph.input[idx].append(inp)
    for idx, out in enumerate(g_2.output):
        if isinstance(fuse_graph.output[idx], str):
            fuse_graph.output[idx] = [fuse_graph.output[idx]]
        fuse_graph.output[idx].append(out)
    return fuse_graph
