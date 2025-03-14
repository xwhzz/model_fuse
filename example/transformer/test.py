import onnxruntime as ort
import onnx
import numpy as np
import argparse
import time
np.random.seed(0)

def run_fuse(model_num: int, index: list[int], data, device):
    so = ort.SessionOptions()
    onnx_model = onnx.load(f'./model/fuse_{model_num}.onnx')
    if device == 'cpu':
        sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
    else:
        sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CUDAExecutionProvider'])
    time_list = []
    for _ in range(10):
        st = time.perf_counter()
        txout = sess.run(None, {
            'hidden_states': data,
            'position_ids': np.concatenate([np.arange(0, 10) for _ in range(data.shape[0])]).reshape(-1, 10).astype(np.int64),
            "info": np.array(index).astype(np.int64)
        })
        time_list.append(time.perf_counter() - st)
    
    print(time_list)
    return txout[0]

def run_model(path: str, start: int, end: int, data, device):
    so = ort.SessionOptions()
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model = onnx.load(path)

    if device == 'cpu':
        sess = ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])
    else:
        sess = ort.InferenceSession(model.SerializeToString(), so, providers=['CUDAExecutionProvider'])
    time_list = [
    ]
    for _ in range(10):
        st = time.perf_counter()
        txout = sess.run(None, {
            'hidden_states': data[start: end],
            'position_ids': np.concatenate([np.arange(0, 10) for _ in range(start, end)]).reshape(-1, 10).astype(np.int64)
        })
        time_list.append(time.perf_counter() - st)
    print(time_list)

    return txout[0]

def test_res(model_num: int, index: list[int], data, device: str):
    model_path = []

    for i in range(1,model_num + 1):
        model_path.append(f'./model/model_{i}.onnx')
    res = []
    cur_index = 0
    for idx, p in enumerate(model_path):
        res.append(run_model(p, cur_index, cur_index + index[idx], data, device))
        cur_index += index[idx]

    res_cat = np.concatenate(res, axis=0)
    fuse_res = run_fuse(model_num, index, data, device)

    if device != "cpu":
        """
        CUDA 存在误差
        """
        assert np.allclose(res_cat, fuse_res, atol=1e-5), (res_cat, fuse_res)
    else:
        assert np.allclose(res_cat, fuse_res), (res_cat, fuse_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    data = np.random.random((2, 10,4096)).astype(np.float32)
    # for i in range(1, 5):
    #   print(i)
    test_res(2, [1,1], data, args.device)

