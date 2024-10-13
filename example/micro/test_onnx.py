import onnxruntime as ort
import onnx
import numpy as np

def run_model():
    so = ort.SessionOptions()

    onnx_model = onnx.load('./model/model_1.onnx')

    sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])

    data = np.random.random((1, 784)).astype(np.float32)

    txout = sess.run(None, {
        'input': data,
        })
    
    print(txout)

run_model()