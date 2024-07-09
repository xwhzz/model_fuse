import onnxruntime as ort
import onnx
import numpy as np
np.random.seed(0)

def run_fuse(model_num: int, index: list[int], data):
    so = ort.SessionOptions()

    so.register_custom_ops_library('./libcustom_op_library.so')

    onnx_model = onnx.load(f'./model/fuse_{model_num}.onnx')

    sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])

    txout = sess.run(None, {
        'Input': data,
        "Info_1": np.array(index).astype(np.int32)
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
    for idx, p in enumerate(model_path):
        res.append(run_model(p, index[idx], index[idx+1], data))

    res_cat = np.concatenate(res, axis=0)

    fuse_res = run_fuse(model_num, index, data)

    assert np.allclose(res_cat, fuse_res)

if __name__ == '__main__':
    
    # Two models test
    data = np.random.random((5,784)).astype(np.float32)
    for i in range(0,6):
        test_res(2, [0, i, 5], data)

    # 5 models test
    data = np.random.random((10,784)).astype(np.float32)
    test_res(5, [0,3,5,7,8,10], data)
