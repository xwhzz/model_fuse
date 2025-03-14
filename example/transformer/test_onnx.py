import onnxruntime as ort
import onnx
import numpy as np


def run_model():
    so = ort.SessionOptions()

    onnx_model = onnx.load('./model/model_1.onnx')
    sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
    for batch_size in range(1,5):
        input_tensor = np.random.random((batch_size,10,4096)).astype(np.float32)
        pos_tensor = np.concatenate([np.arange(10).reshape(1,10)]*batch_size,axis=0).astype(np.int64)
        txout = sess.run(None, {
            'hidden_states': input_tensor,
            'position_ids': pos_tensor
            })
        print(txout)
run_model()