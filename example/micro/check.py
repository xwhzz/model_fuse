import onnx 

model_path = "./model/model_1.onnx"
graph = onnx.load(model_path).graph

# 遍历每个数据流
total_data_flow = len(graph.value_info)
satisfy = 0

remove = []

for node in graph.node:
    if node.op_type == "Constant":
        # print(node.name, node.input, node.output)
        remove.extend(node.output)

for vi in graph.value_info:
    # print(vi.name)
    tensor_shape = str(vi.type.tensor_type.shape.dim)
    # print(tensor_shape)
    if "batch_size" in tensor_shape:
        satisfy += 1
    elif vi.name in remove:
        total_data_flow -= 1
    # else:
    #     print(vi.name,vi.type)

print(total_data_flow, satisfy, round(satisfy / total_data_flow, 2))