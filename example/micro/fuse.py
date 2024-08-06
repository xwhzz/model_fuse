import onnx
import argparse
import netron

from opt.converter import ONNXConverter
from opt.pas import fuse_base, add_op, remove_op, combine


def get_graph(path: str, index: int):
    model = onnx.load(path)
    model = onnx.compose.add_prefix(model, f'{index}_')
    return model.graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=5)

    args = parser.parse_args()

    model_num = args.num
    sep_num = 0
    g_1 = get_graph('./model/model_1.onnx', 0)
    sep_num += len(g_1.node)
    converter = ONNXConverter([g_1.input[0].type], [g_1.output[0].type], model_num)
    g_1 = converter.to_graph(g_1)
    for i in range(1,model_num):
        g = get_graph(f'./model/model_{i+1}.onnx', i)
        sep_num += len(g.node)
        g = converter.to_graph(g)
        g_1 = fuse_base(g_1, g)
    
    add_op(g_1)
    # remove_op(g_1)
    combine(g_1)
    fuse_graph = converter.from_graph(g_1)

    fuse_op_num = len(fuse_graph.node)

    print(f"fuse / sep = {fuse_op_num/sep_num}.")
    converter.export_file(fuse_graph, f'./model/fuse_{model_num}.onnx')
    netron.start(f'./model/fuse_{model_num}.onnx')
