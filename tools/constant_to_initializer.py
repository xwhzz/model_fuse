import onnx
from onnx import numpy_helper
import argparse
import netron

def convert_constant_to_initializer(model: onnx.ModelProto):
    for node in model.graph.node:
        if node.op_type == 'Constant':
            constant_node = node
            constant_tensor = numpy_helper.to_array(constant_node.attribute[0].t)
            new_initializer = numpy_helper.from_array(constant_tensor, name=constant_node.output[0])
            model.graph.initializer.append(new_initializer)
            model.graph.node.remove(constant_node)
    onnx.helper.make_model(model.graph)
    return model

def main(args):
    model = onnx.load(args.input)
    updated_model = convert_constant_to_initializer(model)
    onnx.save(updated_model, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the ONNX model")
    parser.add_argument('--output', type=str, help="Output path")
    args = parser.parse_args()
    main(args)
