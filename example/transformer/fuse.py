import onnx
import argparse
import netron
import sys
sys.path.append("/home/xwh/project/model_fuse")

from opt.converter import ONNXConverter
from opt.opass import fuse_base, add_op, combine


def get_graph(path: str, index: int):
    model = onnx.load(path)
    model = onnx.compose.add_prefix(model, f'{index}_')
    return model.graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=2)

    args = parser.parse_args()

    model_num = args.num

    g_1 = get_graph('./model/model_1.onnx', 0)
    converter = ONNXConverter([g_1.input[0].type, g_1.input[1].type], [g_1.output[0].type], model_num,["hidden_states","position_ids"],["output"],)
    g_1 = converter.to_graph(g_1)
    print('Convert g1 to graph!')
    for i in range(1,model_num):
        g = get_graph(f'./model/model_{i+1}.onnx', i)
        g = converter.to_graph(g)
        print(f'Convert g{i+1} to Graph!')
        g_1 = fuse_base(g_1, g)
        print(f"{i}th fuse complete!")
    add_op(g_1)
    print("Add Op Complete!")
    # remove_op(g_1)
    combine(g_1)
    print("Remove Op Complete!")
    fuse_graph = converter.from_graph(g_1)
    converter.export_file(fuse_graph, f'./model/fu_{model_num}.onnx')
    netron.start(f'./model/fu_{model_num}.onnx')
