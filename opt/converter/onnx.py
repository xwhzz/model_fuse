import onnx.onnx_pb
from ..graph import *
from .base import Converter
import onnx


class ONNXConverter(Converter):
    def __init__(self, input_type: list, ouput_type: list, model_num, input_name: list, output_name: list, ):
        self.input_type = input_type
        self.output_type = ouput_type
        self.model_num = model_num
        self.input_name = input_name
        self.output_name = output_name

        self.gather_list = set()

    def get_node(self, node: onnx.NodeProto, graph: Graph) -> Node:
        node_name = node.name
        node_type = node.op_type
        node_input = list(node.input) # [[inp] for inp in node.input]
        node_output = list(node.output) #[[out] for out in node.output]
        node_other = node.attribute
        node_domain = node.domain
        parameters = []
        new_input = []
        input_index = []
        para_index = []
        constants = []
        constants_index = []
        empty_index = []
        for idx, inp in enumerate(node_input):
            if not inp:
                empty_index.append(idx)
                continue
            if inp not in graph.parameter_list:
                if inp not in graph.constants:
                    new_input.append(inp)
                    input_index.append(idx)
                else:
                    constants_index.append(idx)
                    constants.append(inp)
            else:
                parameters.append(inp)
                para_index.append(idx)
        return Node(node_name, node_type, new_input, node_output, parameters,constants, len(empty_index), node_other,input_index=input_index + para_index + constants_index + empty_index, domain=node_domain)

    def get_parameter(self, init: onnx.TensorProto, graph: onnx.GraphProto, ) -> Parameter:
        tensor_name = init.name
        value = onnx.numpy_helper.to_array(init)
        op_name = ""
        for node in graph.node:
            # if node.name in remove_node:
            #     continue
            for node_input in node.input:
                if tensor_name in node_input:
                    op_name = node.name
                    break
        assert op_name
        return Parameter(tensor_name, value, op_name)

    def str2value(self, name: str, inp: bool, index: int) -> onnx.ValueInfoProto:
        if "info" in name.lower():
            value = onnx.helper.make_tensor_value_info(name,
                        onnx.onnx_pb.TensorProto.INT64, [self.model_num])
        else:
            value = onnx.ValueInfoProto()
            value.name = name
            value.type.CopyFrom(self.input_type[index] if inp else self.output_type[index])

        return value

    def info2tensor(self, para: Parameter) -> onnx.TensorProto:
        tensor = onnx.numpy_helper.from_array(para.value)
        return tensor
    
    @staticmethod
    def create_merge(node_input, node_output, name, axis):
        assert len(node_output) == 1, "Only support one output."
        concat_1 = onnx.helper.make_node('Concat', inputs=node_input, outputs=node_output[0:1], axis=axis, name=name+"_c1")
        return [concat_1,] # + concat_2 + shape_op
        merge = onnx.helper.make_node('Merge', inputs=node_input, outputs=node_output, name=name)
        return [merge]
     
    ## 这里我们需要首先创建几个 Gather节点，这里我们先假设Route节点只有一个
    def info2node(self, node: Node,) -> onnx.NodeProto:
        if not (node.type == "Merge" or node.type == "Route"):
            # print(node.name, node.type, node.inputs, node.parameters, node.constants, node.input_index)
            assert len(node.inputs) + len(node.parameters) + len(node.constants) + node.empty == len(node.input_index)
            index_list = node.input_index
            node_input = [''] * len(node.input_index)
            index = 0
            for inp in node.inputs:
                node_input[index_list[index]] = inp
                index += 1
            for para in node.parameters:
                node_input[index_list[index]] = para
                index += 1
            for constant in node.constants:
                node_input[index_list[index]] = constant
                index += 1
            for _ in range(node.empty):
                node_input[index_list[index]] = ''
                index += 1
        else:
            node_input = node.inputs
        domain = node.domain
        new_node = []
        ## 转化为 split 和 concat 算子
        if node.type == "Route":
            indices = node.gather
            node.type = "Split"
            assert len(node_input) == 1, "Route Op only support one input."
            new_node = [onnx.helper.make_node(node.type, node_input[:1] + [indices], node.outputs, node.name, axis=node.axis)]
            self.gather_list.add(indices)
        elif node.type == "Merge":
            if len(node.inputs) > 1:
                new_node = self.create_merge(node_input, node.outputs, node.name, node.axis)
        else:
            new_node = [onnx.helper.make_node(node.type, node_input, node.outputs, node.name,domain=domain)]
            if node.other:
                new_node[0].attribute.extend(node.other)
        return new_node


    def to_graph(self, model: onnx.GraphProto) -> Graph:
        g = Graph()
    
        g.input.extend([inp.name for inp in model.input])
        g.output.extend([out.name for out in model.output])

        """
        Convert initializer to parameters
        """
        for init in model.initializer:
            param = self.get_parameter(init, model)
            g.add_parameters(param)

        model.value_info.extend(model.output)
        model.value_info.extend(model.input)

        for vi in model.value_info:
            tensor_shape = list(vi.type.tensor_type.shape.dim)
            real_shape = [shape.dim_param if shape.dim_param else shape.dim_value for shape in tensor_shape ]
            edge = Edge(vi.name, real_shape)
            g.add_edge(edge)

        for node in model.node:
            if node.op_type == "Constant":
                tensor_name = node.output[0]
                constant_tensor = onnx.numpy_helper.to_array(node.attribute[0].t)
                g.add_constant(tensor_name, constant_tensor)

        for node in model.node:
            nod = self.get_node(node, g)
            g.add_node(nod,)

        return g

    def from_graph(self, graph: Graph, num: int=2) -> onnx.GraphProto:
        g = onnx.GraphProto()
        g.name = 'Fused model'

        mp = {}
        for idx, inp in enumerate(graph.input):
            if idx != len(graph.input) - 1:
                mp[inp] = self.input_name[idx]
            else:
                mp[inp] = inp
        for idx, out in enumerate(graph.output):
            mp[out] = self.output_name[idx]

        g.input.extend([self.str2value(mp[inp], True, idx) for idx, inp in enumerate(graph.input)])
        g.output.extend([self.str2value(mp[out], False, idx) for idx, out in enumerate(graph.output)])

        for name, p in graph.parameter_list.items():
            tensor = self.info2tensor(p)
            tensor.name = name
            g.initializer.append(tensor)
        
        for name, node in graph.node_list.items():
            node.update(mp)
            g.node.extend(self.info2node(node))

        index = 0
        for name, constant in graph.constants.items():
            g.node.extend([onnx.helper.make_node('Constant', [], [name], name=f"Constant_{index}", value=onnx.numpy_helper.from_array(constant))])
            index += 1
        """
        FIXME: 后续可能需要重写
        """
        for gather in self.gather_list:
            l = eval(gather.split('_')[-1])
            if len(l) == self.model_num:
                g.node.extend([onnx.helper.make_node('Identity', ["info"], [gather], name=gather)])
            else:
                constant = np.array(l).astype(np.int64)
                name = f"{gather}_constant"
                g.node.extend([onnx.helper.make_node('Constant', [], [name], name=name, value=onnx.numpy_helper.from_array(constant))])
                g.node.extend([onnx.helper.make_node('Gather', ["info", name], [gather], name=gather)])

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

    def export_file(self, graph: onnx.GraphProto, file_name: str='fused_model.onnx', large: bool = False):
        model = onnx.helper.make_model(graph)
        model.opset_import[0].version = 15
        if large:
            onnx.save_model(model, file_name, save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024, convert_attribute=False)
        else:
            onnx.save(model, file_name)
