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
        input_index = []
        para_index = []
        for idx, inp in enumerate(node_input):
            if graph.name2para.get(inp, None) is None:
                new_input.append(inp)
                input_index.append(idx)
            else:
                parameters.append(inp)
                para_index.append(idx)
        return NodeInfo(node_type, new_input, node_output, parameters, node_other,InputIndex=input_index + para_index), node_name

    def get_parameter(self, init: onnx.TensorProto, graph: onnx.GraphProto, remove_node: set[str] | None=None) -> tuple[ParameterInfo, str]:
        tensor_name = init.name

        value = onnx.numpy_helper.to_array(init)
        op_name = ""
        for node in graph.node:
            if remove_node is not None:
                if node.name in remove_node:
                    continue
            for node_input in node.input:
                if tensor_name in node_input:
                    op_name = node.name
                    break
        assert op_name
        return ParameterInfo.get_info(value, op_name), tensor_name

    def str2value(self, name: str, inp: bool, index: int) -> onnx.ValueInfoProto:
        if "info" in name.lower():
            value = onnx.helper.make_tensor_value_info(name,
                        onnx.onnx_pb.TensorProto.INT64, [self.num ])
        else:
            value = onnx.ValueInfoProto()
            value.name = name
            value.type.CopyFrom(self.input_type[index] if inp else self.output_type[index])

        return value

    def info2tensor(self, para: ParameterInfo) -> onnx.TensorProto:
        tensor = onnx.numpy_helper.from_array(para.value)

        return tensor

    def info2node(self, node: NodeInfo, name: str) -> onnx.NodeProto:
        if not (node.Type == "Merge" or node.Type == "Route"):
            assert len(node.Input) + len(node.Parameters) == len(node.InputIndex)
            index_list = node.InputIndex
            node_input = [''] * len(node.InputIndex)
            index = 0
            for inp in node.Input:
                node_input[index_list[index]] = inp
                index += 1
            for para in node.Parameters:
                node_input[index_list[index]] = para
                index += 1
        else:
            node_input = node.Input
        domain = None
        ## 自定义算子
        if node.Type == "Route":
            node.Type = "Split"
        elif node.Type == "Merge":
            # node.Type = "Concat"
            domain = "test.customop"
            # node.Other = [onnx.helper.make_attribute("is_merge", 1), onnx.helper.make_attribute("axis", 0)]
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

        for vi in model.value_info:
            try:
                # print(vi.name)
                g.name2shape[vi.name] = vi.type.tensor_type.shape.dim[0].dim_param == "batch_size"
            except:
                g.name2shape[vi.name] = False
        for inp in model.input:
            g.name2shape[inp.name] = inp.type.tensor_type.shape.dim[0].dim_param == "batch_size"
        for out in model.output:
            g.name2shape[out.name] = out.type.tensor_type.shape.dim[0].dim_param == "batch_size"

        remove_node = set()
        for node in model.node:
            if node.op_type == 'Constant':
                constant_tensor = onnx.numpy_helper.to_array(node.attribute[0].t)
                tensor_name = node.output[0]
                new_initializer = onnx.numpy_helper.from_array(constant_tensor, name=tensor_name)
                remove_node.add(tensor_name)
                g.add_parameters(*(self.get_parameter(new_initializer, model, remove_node)))
                g.add_constant(tensor_name)
            else:
                nod, name = self.get_node(node, g)
                g.add_node(nod, name)

        return g

    def from_graph(self, graph: Graph, num: int=2) -> onnx.GraphProto:
        g = onnx.GraphProto()
        g.name = 'Fused model'

        g.input.extend([self.str2value(inp, True, idx) for idx, inp in enumerate(graph.input)])
        g.output.extend([self.str2value(out, False, idx) for idx, out in enumerate(graph.output)])

        for para in graph.paramter_list.values():
            for name, p in para.items():
                tensor = self.info2tensor(p)
                tensor.name = name
                g.initializer.append(tensor)
        
        for name, node in graph.node_list.items():
            g.node.extend([self.info2node(node, name)])
            
        self.clean_unused_initializers(g)
        self.clean_unused_node(g)
        return g


    def clean_unused_initializers(self, graph: onnx.GraphProto):
        print('Clean unused initializer Start!')
        all_initializers = set(initializer.name for initializer in graph.initializer)
        input_names = set(input.name for input in graph.input)

        used_initializers = set()
        for node in graph.node:
            used_initializers.update(node.input)

        unused_initializers = all_initializers - (used_initializers | input_names)

        for initializer_name in unused_initializers:
            graph.initializer.remove(next(initializer for initializer in graph.initializer if initializer.name == initializer_name))
        
        print('Clean unused initializers Complete!')

    def clean_unused_node(self, graph: onnx.GraphProto):
        print('Clean unused node Start!')
        has_change = True
        while has_change:
            has_change = False
            all_input = set()
            for node in graph.node:
                all_input.update(node.input)

            all_input = all_input | set(output.name for output in graph.output)
            node_list = []
            for node in graph.node:
                if all_input.isdisjoint(node.output):
                    has_change = True
                    node_list.append(node)
            for node in node_list:
                graph.node.remove(node)
        print('Clean unused node Complete!')

    @staticmethod
    def export_file(graph: onnx.GraphProto, file_name: str='fused_model.onnx'):
        model = onnx.helper.make_model(graph)
        model.opset_import[0].version = 14
        onnx.save(model, file_name)



