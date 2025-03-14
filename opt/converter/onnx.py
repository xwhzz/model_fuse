import onnx.onnx_pb
from ..graph import *
from .base import Converter
import onnx


import numpy as np
import onnx
import pickle

class ONNXConverter(Converter):
    def __init__(self, input_type: list, ouput_type: list, model_num, input_name: list, output_name: list, ):
        self.input_type = input_type
        self.output_type = ouput_type
        self.model_num = model_num
        self.input_name = input_name
        self.output_name = output_name

        self.model_type = "unet"
        self.gather_list = set()

    def get_node(self, node: onnx.NodeProto, graph: Graph, node_map = None) -> Node:
        need_diff = [f"/text_model/encoder/layers.11/self_attn/{s}_proj/default_0/MatMul" for s in ["q", "k", "v", "out"]] + \
            [f"/text_model/encoder/layers.11/self_attn/{s}_proj/default_0_1/MatMul" for s in ["q", "k", "v", "out"]] + \
            [f"/text_model/encoder/layers.11/mlp/fc{i}/default_0{j}/MatMul" for i in [1,2] for j in ["", "_1"]]

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
        ad = False
        if node_map:
            for idx, inp in enumerate(node_input):
                if inp in node_map:
                    node_input[idx] = node_map[inp]
                    ad = True
        for nd in need_diff:
            if nd in node_name:
                ad = True
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
        return Node(node_name, node_type, new_input, node_output, parameters,constants, len(empty_index), node_other,input_index=input_index + para_index + constants_index + empty_index, domain=node_domain, ad=ad)

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
                        onnx.onnx_pb.TensorProto.INT64, [None])
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


    def fuse_multi_lora(self, graph: Graph):
        num = self.model_num
        expect_num = 2 * num + 2
        # cur_num = 0
        new_node_list = []
        new_init_list = []
        ranks = [32] * num
        a_len = np.array([r * 4 for r in ranks], dtype=np.int64)
        a_start = np.zeros_like(a_len)
        a_start[1:] = np.cumsum(a_len[:-1], axis=0)
        a_loc = np.arange(sum(ranks) * 4, dtype=np.int64)
        a_scaling = np.array([1] * num, dtype=np.float16)

        a_len = onnx.numpy_helper.from_array(a_len, "a_len")
        a_start = onnx.numpy_helper.from_array(a_start, "a_start")
        a_loc = onnx.numpy_helper.from_array(a_loc, "a_loc")
        a_scaling = onnx.numpy_helper.from_array(a_scaling, "a_scaling")
        
        b_len = np.array([r for r in ranks], dtype=np.int64)
        b_start = np.zeros_like(b_len)
        b_start[1:] = np.cumsum(b_len[:-1], axis=0)
        b_loc = np.arange(sum(ranks), dtype=np.int64)
        b_scaling = np.array([1] * num, dtype=np.float16)
        b_len = onnx.numpy_helper.from_array(b_len, "b_len")
        b_start = onnx.numpy_helper.from_array(b_start, "b_start")
        b_loc = onnx.numpy_helper.from_array(b_loc, "b_loc")
        b_scaling = onnx.numpy_helper.from_array(b_scaling, "b_scaling")
        new_init_list.extend([a_len, a_start, a_loc, a_scaling, b_len, b_start, b_loc, b_scaling])
        batch_info = "info" 
        qkvo_map = {s: i for i, s in enumerate(["q", "k", "v", "out"])}
        remove_par = set()
        for i in range(0,12):
            par_0 = []
            par_1 = []
            par_name_0 = f"bgmv_weight_0_{i}"
            par_name_1 = f"bgmv_weight_1_{i}"
            for s in ["q", "k", "v", "out"]:
                cur_num = 0
                item = f"text_model/encoder/layers.{i}/self_attn/{s}_proj/"
                node_0 = []
                node_1 = []
                node_split = []
                node_merge = []
                for name, node in graph.node_list.items():
                    if node.type == "MatMul":
                        if item + 'default_0/MatMul' in name:
                           node_0.append(node)
                           cur_num += 1
                        if item + 'default_0_1/MatMul' in name:
                            node_1.append(node)
                            cur_num += 1
                    if node.type == "Route":
                        if item in node.name:
                            node_split.append(node)
                            cur_num += 1
                    if node.type == "Merge":
                        if item in node.name:
                            node_merge.append(node)
                            cur_num += 1
                    if cur_num == expect_num:
                        break
                if cur_num != expect_num:
                    raise 
                # print([node.name for node in node_0+node_1+node_split+node_merge])
                new_node_input = node_split[0].inputs[0]
                new_node_output = node_merge[0].outputs[0]

                for node in node_0:
                    par_0.append(node.parameters[0])
                for node in node_1:
                    par_1.append(node.parameters[0])
                new_node = onnx.helper.make_node('CustomOpBGMV', inputs=[new_node_input, 
                                                              par_name_0,
                                                              batch_info,
                                                              'a_len',
                                                              'a_scaling',
                                                              # 'a_loc',
                                                              'a_start',
                                                              'a_loc',
                                                              ],
                                                    outputs = [f'bgmv_out_{i}_{s}'],
                                                    name=f'bgmv_0_{i}_{s}',
                                                    qkvo=qkvo_map[s],
                                                    qkvon=4,
                                                    domain="test.customop")
                new_node_1 = onnx.helper.make_node('CustomOpBGMV', inputs=[f'bgmv_out_{i}_{s}', 
                                                              par_name_1,
                                                              batch_info,
                                                              'a_len',
                                                              'a_scaling',
                                                              # 'a_loc',
                                                              'a_start',
                                                              'a_loc',
                                                              ],
                                                    outputs = [new_node_output],
                                                    name=f'bgmv_1_{i}_{s}',
                                                    qkvo=qkvo_map[s],
                                                    qkvon=4,
                                                    domain="test.customop")
                new_node_list.extend([new_node, new_node_1])
                ## 去除多余节点
                for node in node_0 + node_1 + node_split + node_merge:
                    del graph.node_list[node.name]
            assert len(par_0) == len(par_1) == num * 4, "Parameter number error!"
            new_par_0 = []
            new_par_1 = []
            for idx in range(num):
                for j in range(4):
                    # new_par_0.append(par_0[idx * 4 + j])
                    # new_par_1.append(par_1[idx * 4 + j])
                    new_par_0.append(par_0[idx + j * 4])
                    new_par_1.append(par_1[idx + j * 4])

            par_array_0 = np.concatenate([graph.parameter_list[p].value.T for p in new_par_0])
            new_par_1_ = []
            for p in new_par_1:
                cur_tensor = graph.parameter_list[p].value
                R, H = cur_tensor.shape
                cur_tensor = cur_tensor.T.reshape((R, H))
                new_par_1_.append(cur_tensor)
            par_array_1 = np.concatenate(new_par_1_)
            tensor_0 = onnx.numpy_helper.from_array(par_array_0, par_name_0)
            tensor_1 = onnx.numpy_helper.from_array(par_array_1, par_name_1)
            new_init_list.extend([tensor_0, tensor_1])

            ## 清理多余的参数
            # for par in new_par_0 + new_par_1:
            remove_par.update(new_par_0 + new_par_1)
                
            
            for s in ["fc1", "fc2"]:
                cur_num = 0
                item = f"text_model/encoder/layers.{i}/mlp/{s}/"
                node_0 = []
                node_1 = []
                node_split = []
                node_merge = []
                par_0 = []
                par_1 = []
                for name, node in graph.node_list.items():
                    # s = item
                    if node.type == "MatMul":
                        if item + 'default_0/MatMul' in name:
                           node_0.append(node)
                           cur_num += 1
                        if item + 'default_0_1/MatMul' in name:
                            node_1.append(node)
                            cur_num += 1
                    if node.type == "Route":
                        if item in node.name:
                            node_split.append(node)
                            cur_num += 1
                    if node.type == "Merge":
                        if item in node.name:
                            node_merge.append(node)
                            cur_num += 1
                    if cur_num == expect_num:
                        break
                if cur_num != expect_num:
                    raise 
                new_node_input = node_split[0].inputs[0]
                new_node_output = node_merge[0].outputs[0]

                for node in node_0:
                    par_0.append(node.parameters[0])
                for node in node_1:
                    par_1.append(node.parameters[0])
                assert len(par_0) == len(par_1) == num, "Parameter number error!"
                par_array_0 = np.concatenate([graph.parameter_list[p].value.T for p in par_0])
                tensor_0 = onnx.numpy_helper.from_array(par_array_0, par_0[0])
                # par_array_1 = np.concatenate([graph.parameter_list[p].value for p in par_1])
                # R, H = par_array_1.shape
                # par_array_1 = par_array_1.T.reshape((R, H))
                new_par_1 = []
                for p in par_1:
                    cur_tensor = graph.parameter_list[p].value
                    R, H = cur_tensor.shape
                    cur_tensor = cur_tensor.T.reshape((R, H))
                    new_par_1.append(cur_tensor)
                par_array_1 = np.concatenate(new_par_1)
                tensor_1 = onnx.numpy_helper.from_array(par_array_1, par_1[0])
                new_node = onnx.helper.make_node('CustomOpBGMV', inputs=[new_node_input, 
                                                              tensor_0.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              # 'b_loc',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [f'bgmv_out_{i}_{s}'],
                                                    name=f'bgmv_0_{i}_{s}',
                                                    qkvo=0,
                                                    qkvon = 1,
                                                    domain="test.customop")
                new_node_1 = onnx.helper.make_node('CustomOpBGMV', inputs=[f'bgmv_out_{i}_{s}', 
                                                              tensor_1.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              # 'b_loc',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [new_node_output],
                                                    name=f'bgmv_1_{i}_{s}',
                                                    qkvo=0,
                                                    qkvon=1,
                                                    domain="test.customop")
                new_node_list.extend([new_node, new_node_1])
                new_init_list.extend([tensor_0, tensor_1])

                for node in node_0 + node_1 + node_split + node_merge:
                    del graph.node_list[node.name]
                remove_par.update(par_0 + par_1)
        for par in remove_par:
            del graph.parameter_list[par]
        return new_node_list, new_init_list

    def fuse_multi_lora_unet(self, graph: Graph):
        num = self.model_num
        expect_num = 2 * num + 2
        # cur_num = 0
        new_node_list = []
        new_init_list = []
        ranks = [32] * num
        
        b_len = np.array([r for r in ranks], dtype=np.int64)
        b_start = np.zeros_like(b_len)
        b_start[1:] = np.cumsum(b_len[:-1], axis=0)
        b_loc = np.arange(sum(ranks), dtype=np.int64)
        b_scaling = np.array([1] * num, dtype=np.float16)
        b_len = onnx.numpy_helper.from_array(b_len, "b_len")
        b_start = onnx.numpy_helper.from_array(b_start, "b_start")
        b_loc = onnx.numpy_helper.from_array(b_loc, "b_loc")
        b_scaling = onnx.numpy_helper.from_array(b_scaling, "b_scaling")
        new_init_list.extend([b_len, b_start, b_loc, b_scaling])
        # batch_info = "info" 
        remove_par = set()
        conv_lora = [
            f"/down_blocks.{i}/attentions.{j}/{s}" for i in range(3) for j in range(2) for s in ["proj_in", "proj_out"]
        ] + [
            f"/mid_block/attentions.0/{s}" for s in ["proj_in", "proj_out"]
        ] + [
            f"/up_blocks.{i}/attentions.{j}/{s}" for i in range(1,4) for j in range(3) for s in ["proj_in", "proj_out"]
        ]

        """
        1_/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q/default_0/MatMul
        """
        matmul_lora = [
            f"/down_blocks.{i}/attentions.{j}/transformer_blocks.0/{s}" for i in range(3) for j in range(2) for s in ["attn1/to_q", "attn1/to_k", "attn1/to_v", "attn1/to_out.0", "attn2/to_q", "attn2/to_k", "attn2/to_v", "attn2/to_out.0", "ff/net.0/proj", "ff/net.2"]
        ] + [
            f"/mid_block/attentions.0/transformer_blocks.0/{s}" for s in ["attn1/to_q", "attn1/to_k", "attn1/to_v", "attn1/to_out.0", "attn2/to_q", "attn2/to_k", "attn2/to_v", "attn2/to_out.0","ff/net.0/proj", "ff/net.2"]
        ] + [
            f"/up_blocks.{i}/attentions.{j}/transformer_blocks.0/{s}" for i in range(1,4) for j in range(3) for s in ["attn1/to_q", "attn1/to_k", "attn1/to_v", "attn1/to_out.0", "attn2/to_q", "attn2/to_k", "attn2/to_v", "attn2/to_out.0","ff/net.0/proj", "ff/net.2"]
        ]
        info_cnt = 1
        info_map = {}
        if True:
            for idx, item in enumerate(matmul_lora):
                cur_num = 0
                node_0 = []
                node_1 = []
                node_split = []
                node_merge = []
                par_0 = []
                par_1 = []
                for name, node in graph.node_list.items():
                    # s = item
                    if node.type == "MatMul":
                        if item + '/default_0/MatMul' in name:
                           node_0.append(node)
                           cur_num += 1
                        if item + '/default_0_1/MatMul' in name:
                            node_1.append(node)
                            cur_num += 1
                    if node.type == "Route":
                        if item in node.name:
                            node_split.append(node)
                            cur_num += 1
                    if node.type == "Merge":
                        if item in node.name:
                            node_merge.append(node)
                            cur_num += 1
                    if cur_num == expect_num:
                        break
                if cur_num != expect_num:
                    # print(item, cur_num)
                    # print([node.name for node in node_0+node_1+node_split+node_merge])
                    # continue
                    raise

                new_node_input = node_split[0].inputs[0]
                new_node_output = node_merge[0].outputs[0]
                input_shape = graph.edge_list[node_0[0].inputs[0]].shape
                if len(input_shape) == 3:
                    info_shape = input_shape[1]
                    if info_shape in info_map:
                        batch_info = info_map[info_shape]
                    else:
                        batch_info = f"info_{info_cnt}"
                        info_map[info_shape] = batch_info
                        info_cnt += 1

                for node in node_0:
                    par_0.append(node.parameters[0])
                for node in node_1:
                    par_1.append(node.parameters[0])
                assert len(par_0) == len(par_1) == num, "Parameter number error!"
                par_array_0 = np.concatenate([graph.parameter_list[p].value.T for p in par_0])
                tensor_0 = onnx.numpy_helper.from_array(par_array_0, par_0[0])
                new_par_1 = []
                for p in par_1:
                    cur_tensor = graph.parameter_list[p].value
                    R, H = cur_tensor.shape
                    cur_tensor = cur_tensor.T.reshape((R, H))
                    new_par_1.append(cur_tensor)
                par_array_1 = np.concatenate(new_par_1)
                tensor_1 = onnx.numpy_helper.from_array(par_array_1, par_1[0])
                new_node = onnx.helper.make_node('CustomOpBGMV', inputs=[new_node_input, 
                                                              tensor_0.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [f'bgmv_out_{idx}'],
                                                    name=f'bgmv_0_{idx}',
                                                    qkvo=0,
                                                    qkvon = 1,
                                                    domain="test.customop")
                new_node_1 = onnx.helper.make_node('CustomOpBGMV', inputs=[f'bgmv_out_{idx}', 
                                                              tensor_1.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [new_node_output],
                                                    name=f'bgmv_1_{idx}',
                                                    qkvo=0,
                                                    qkvon=1,
                                                    domain="test.customop")
                new_node_list.extend([new_node, new_node_1])
                new_init_list.extend([tensor_0, tensor_1])

                for node in node_0 + node_1 + node_split + node_merge:
                    del graph.node_list[node.name]
                remove_par.update(par_0 + par_1)
            idxx = idx
        # if False:
            for idx, item in enumerate(conv_lora, start=idxx+1):
                cur_num = 0
                node_0 = []
                node_1 = []
                node_split = []
                node_merge = []
                par_0 = []
                par_1 = []
                for name, node in graph.node_list.items():
                    # s = item
                    if node.type == "Conv":
                        if item + '/default_0/Conv' in name:
                           node_0.append(node)
                           cur_num += 1
                        if item + '/default_0_1/Conv' in name:
                            node_1.append(node)
                            cur_num += 1
                    if node.type == "Route":
                        if item in node.name:
                            node_split.append(node)
                            cur_num += 1
                    if node.type == "Merge":
                        if item in node.name:
                            node_merge.append(node)
                            cur_num += 1
                    if cur_num == expect_num:
                        break
                if cur_num != expect_num:
                    print(item, cur_num)
                    print([node.name for node in node_0+node_1+node_split+node_merge])
                    continue
                new_node_input = node_split[0].inputs[0]
                new_node_output = node_merge[0].outputs[0]
                input_shape = graph.edge_list[node_0[0].inputs[0]].shape
                if len(input_shape) == 4:
                    info_shape = input_shape[2] * input_shape[3]
                    if info_shape in info_map:
                        batch_info = info_map[info_shape]
                    else:
                        batch_info = f"info_{info_cnt}"
                        info_map[info_shape] = batch_info
                        info_cnt += 1
                for node in node_0:
                    par_0.append(node.parameters[0])
                for node in node_1:
                    par_1.append(node.parameters[0])
                assert len(par_0) == len(par_1) == num, "Parameter number error!"
                par_array_0 = np.concatenate([np.squeeze(graph.parameter_list[p].value, axis=(2,3)).transpose(1,0).T for p in par_0])
                tensor_0 = onnx.numpy_helper.from_array(par_array_0, par_0[0])
                new_par_1 = []
                for p in par_1:
                    cur_tensor = np.squeeze(graph.parameter_list[p].value, axis=(2,3)).transpose(1,0)
                    R, H = cur_tensor.shape
                    cur_tensor = cur_tensor.T.reshape((R, H))
                    new_par_1.append(cur_tensor)
                par_array_1 = np.concatenate(new_par_1)
                tensor_1 = onnx.numpy_helper.from_array(par_array_1, par_1[0])
                in_node = onnx.helper.make_node('Transpose',
                    inputs=[new_node_input],
                    outputs=[f'{new_node_input}_t'],
                    name=f'{new_node_input}_tt',
                    perm=[0, 2, 3, 1])
                
                new_node = onnx.helper.make_node('CustomOpBGMV', inputs=[new_node_input+"_t", 
                                                              tensor_0.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [f'bgmv_out_{idx}'],
                                                    name=f'bgmv_0_{idx}',
                                                    qkvo=0,
                                                    qkvon = 1,
                                                    domain="test.customop")
                new_node_1 = onnx.helper.make_node('CustomOpBGMV', inputs=[f'bgmv_out_{idx}', 
                                                              tensor_1.name,
                                                              batch_info,
                                                              'b_len',
                                                              'b_scaling',
                                                              'b_start',
                                                              'b_loc',
                                                              ],
                                                    outputs = [new_node_output+"_t"],
                                                    name=f'bgmv_1_{idx}',
                                                    qkvo=0,
                                                    qkvon=1,
                                                    domain="test.customop")
                out_node = onnx.helper.make_node('Transpose',
                    inputs=[new_node_output+"_t"],
                    outputs=[new_node_output],
                    name=f'{new_node_output}_tt',
                    perm=[0, 3, 1, 2])
                new_node_list.extend([new_node, new_node_1, in_node, out_node])
                new_init_list.extend([tensor_0, tensor_1])

                for node in node_0 + node_1 + node_split + node_merge:
                    del graph.node_list[node.name]
                remove_par.update(par_0 + par_1)        
        
        for par in remove_par:
            del graph.parameter_list[par]
        print(info_map)
        return new_node_list, new_init_list, info_map

    def to_graph(self, model: onnx.GraphProto) -> Graph:
        if self.model_type == "text_encoder":
            remove_node = [f"Identity_{i}" for i in range(666, 670)]
        else:
            remove_node = []
        def in_remove_node(name):
            return any((node in name for node in remove_node))
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
                
        node_map = {}
        for node in model.node:
            if node.op_type == "Identity" and in_remove_node(node.name):
                node_map[node.output[0]] = node.input[0]
        # print(node_map)
        for node in model.node:  
            if node.op_type == "Identity" and in_remove_node(node.name):
                continue    
            nod = self.get_node(node, g, node_map)
            g.add_node(nod,)

        return g

    def from_graph(self, graph: Graph, num: int=2) -> onnx.GraphProto:
        g = onnx.GraphProto()
        g.name = 'Fused model'
        if self.model_type == "text_encoder":
            new_node_list, new_init_list = self.fuse_multi_lora(graph)
            g.node.extend(new_node_list)
            g.initializer.extend(new_init_list)
        elif self.model_type == "unet":
            new_node_list, new_init_list, info_map = self.fuse_multi_lora_unet(graph)
            g.node.extend(new_node_list)
            g.initializer.extend(new_init_list)
            info_list = list(info_map.values())
            # info_list.remove("info")
        mp = {}
        for idx, inp in enumerate(graph.input):
            if idx != len(graph.input) - 1:
                mp[inp] = self.input_name[idx]
            else:
                mp[inp] = inp
        for idx, out in enumerate(graph.output):
            mp[out] = self.output_name[idx]

        g.input.extend([self.str2value(mp[inp], True, idx) for idx, inp in enumerate(graph.input)])
        for info in info_list:
            g.input.extend([self.str2value(info, True, 0)])
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

        np.random.random()
