from opt.graph import *

def fuse_base(g_1: Graph, g_2: Graph) -> Graph:
    fuse_graph = g_1
    """
    可以fuse constant的节点
    """
    for node_name, node_info in g_2.node_list.items():
        has_same = False
        if node_info.has_weight():
            same_ps = set()
            has_same = True
            for idx, para in enumerate(node_info.Parameters):
                same = set()
                para_info = g_2.paramter_list[g_2.name2para[para]][para]
                try:
                    for fuse_para in fuse_graph.paramter_list[para_info.hash].values():
                        if para_info.value.shape == fuse_para.value.shape and np.allclose(para_info.value, fuse_para.value):
                            same.add(fuse_para.node)
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
                node_fuse.Input.append(node_info.Input[0])
                node_fuse.Output.append(node_info.Output[0])
                # node_fuse.num += 1
        if not has_same:   
            fuse_graph.node_list[node_name] = node_info
            for para in node_info.Parameters:
                para_info = g_2.paramter_list[g_2.name2para[para]][para]
                # fuse_graph.paramter_list[para_info.hash] = {}#[para] = para_info
                if para_info.hash not in fuse_graph.paramter_list:
                    fuse_graph.paramter_list[para_info.hash] = {}
                fuse_graph.paramter_list[para_info.hash][para] = para_info
                fuse_graph.name2para[para] = para_info.hash
    
    fuse_graph.input.extend(g_2.input)
    fuse_graph.output.extend(g_2.output)
    
    return fuse_graph
