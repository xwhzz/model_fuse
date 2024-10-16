import onnxruntime as ort
import onnx
import numpy as np
np.random.seed(0)

def run_fuse(model_num: int, index: list[int], data):
    so = ort.SessionOptions()

    # so.register_custom_ops_library('../libcustom_op_library.so')

    onnx_model = onnx.load(f'./model/fuse_{model_num}.onnx')

    sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])

    txout = sess.run(None, {
        'input': data,
        "info": np.array(index).astype(np.int64)
        })
    
    return txout[0]

def run_model(path: str, start: int, end: int, data):
    so = ort.SessionOptions()
    model = onnx.load(path)

    sess = ort.InferenceSession(model.SerializePartialToString(),so,providers=['CPUExecutionProvider'])

    txout = sess.run(None, {
        'input': data[start: end]
    })

    return txout[0]

def test_res(model_num: int, index: list[int], data):
    model_path = []
    for i in range(1,model_num+1):
        model_path.append(f'./model/model_{i}.onnx')
    res = []
    cur_index = 0
    for idx, p in enumerate(model_path):
        res.append(run_model(p, cur_index, cur_index + index[idx], data))
        cur_index += index[idx]

    res_cat = np.concatenate(res, axis=0)
    fuse_res = run_fuse(model_num, index, data)

    assert np.allclose(res_cat, fuse_res), (res_cat, fuse_res)

if __name__ == '__main__':
    
    # Two models test
    data = np.random.random((5,784)).astype(np.float32)
    for i in range(0,6):
        test_res(2, [i, 5-i], data)

    # # 5 models test
    data = np.random.random((10,784)).astype(np.float32)
    test_res(5, [2,3,1,2,2], data)
