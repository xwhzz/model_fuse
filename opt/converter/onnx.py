import onnx.onnx_pb
from opt.graph import *
from opt.converter.base import Converter
import onnx


class ONNXConverter(Converter):
    def __init__(self, input_type, ouput_type, num):
        self.input_type = input_type
        self.output_type = ouput_type
        self.num = num

    def remove_identity(self, graph: onnx.GraphProto):
        for node in graph.node:
            if node.op_type == 'Identity':
                node_input = node.input[0]
                node_output = node.output[0]
                add_init = False
                if ("weight" in node_input or "bias" in node_input) \
                    and ("weight" in node_output or "bias" in node_output):
                    for init in graph.initializer:
                        if init.name == node_input:
                            tensor = onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(init))
                            tensor.name = node_output
                            graph.initializer.append(tensor)
                            add_init = True
                            break
                if add_init:
                    graph.node.remove(node)

    def get_node(self, node: onnx.NodeProto, graph: Graph):
        node_name = node.name
        node_type = node.op_type
        node_input = list(node.input)
        node_output = list(node.output)
        node_other = node.attribute

        parameters = []
        new_input = []
        for inp in node_input:
            if graph.name2para.get(inp, None) is None:
                new_input.append(inp)
            else:
                parameters.append(inp)

        return NodeInfo(node_type, new_input, node_output, parameters, node_other), node_name

    def get_parameter(self, init: onnx.TensorProto, graph: onnx.GraphProto) -> ParameterInfo:
        tensor_name = init.name

        value = onnx.numpy_helper.to_array(init)
        op_name = ""
        for node in graph.node:
            for node_input in node.input:
                if tensor_name in node_input:
                    op_name = node.name
                    break
        assert op_name
        return ParameterInfo.get_info(value, op_name), tensor_name

    def str2value(self, name: str, inp: bool) -> onnx.ValueInfoProto:
        if "info" in name.lower():
            value = onnx.helper.make_tensor_value_info(name,
                        onnx.onnx_pb.TensorProto.INT32, [self.num + 1])
        else:
            value = onnx.ValueInfoProto()
            value.name = name
            value.type.CopyFrom(self.input_type if inp else self.output_type)

        return value

    def info2tensor(self, para: ParameterInfo) -> onnx.TensorProto:
        tensor = onnx.numpy_helper.from_array(para.value)

        return tensor

    def info2node(self, node: NodeInfo, name: str) -> onnx.NodeProto:
        node_input = node.Input + node.Parameters
        domain = None
        ## 自定义算子
        if node.Type == "Merge" or node.Type == "Route":
            # domain = "ai.onnx.contrib"
            domain = "test.customop"
            # new_node = onnx.helper.make_node(node.Type, node_input, node.Output, name, domain=domain, num=node.num)
        # else:
        new_node = onnx.helper.make_node(node.Type, node_input, node.Output, name, domain=domain)
        if node.Other:
            new_node.attribute.extend(node.Other)
        return new_node


    def to_graph(self, model: onnx.GraphProto) -> Graph:
        g = Graph()
    
        g.input.extend([inp.name for inp in model.input])
        g.output.extend([out.name for out in model.output])


        for init in model.initializer:
            param, name = self.get_parameter(init, model)
            g.add_parameters(param, name)

        for node in model.node:
            nod, name = self.get_node(node, g)
            g.add_node(nod, name)
        return g

    def from_graph(self, graph: Graph, num: int=2) -> onnx.GraphProto:
        g = onnx.GraphProto()
        g.name = 'Fused model'

        g.input.extend([self.str2value(inp, True) for inp in graph.input])
        g.output.extend([self.str2value(out, False) for out in graph.output])

        for para in graph.paramter_list.values():
            for name, p in para.items():
                tensor = self.info2tensor(p)
                tensor.name = name
                g.initializer.append(tensor)
        
        for name, node in graph.node_list.items():
            g.node.extend([self.info2node(node, name)])

        return g
    
    @staticmethod
    def export_file(graph: onnx.GraphProto, file_name: str='fused_model.onnx'):
        model = onnx.helper.make_model(graph)
        model.opset_import[0].version = 14
        onnx.save(model, file_name)

