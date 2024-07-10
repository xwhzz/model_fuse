import onnxruntime as ort
import onnx
import numpy as np
np.random.seed(0)

def run_fuse(model_num: int, index: list[int], data):
    so = ort.SessionOptions()

    so.register_custom_ops_library('../libcustom_op_library.so')

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
    # print(res_cat)
    # print(fuse_res)

    """
    [[[ 4.7617319e-01  7.5242633e-01  5.9879410e-01 ...  5.0899351e-01
    2.6455647e-01  5.1510149e-01]
  [ 2.5535122e-01  6.8977261e-01  4.1920286e-01 ...  7.6484400e-01
    6.3488644e-01  6.4388931e-01]
  [ 4.1237405e-01  1.7340299e-01  6.7449671e-01 ...  2.1270144e-01
    6.3466752e-01  8.6857826e-01]
  ...
  [ 8.1065603e-02  6.6501498e-02  5.6959128e-01 ...  5.4045892e-01
    8.6184597e-01  9.3150407e-01]
  [-7.6004758e-02  5.4888248e-01  8.7145883e-01 ...  6.0898119e-01
    6.4421809e-01  8.7105870e-01]
  [ 4.6014377e-01  9.1836774e-01 -5.3356197e-03 ...  4.1441053e-01
    7.9786223e-01  8.9707732e-01]]

 [[ 5.6181866e-01  9.2686254e-01  3.0097404e-02 ... -3.8611218e-03
    4.7412285e-01  4.4197452e-01]
  [ 5.8265102e-01  7.3647022e-01  7.0062941e-01 ...  6.4639229e-01
    7.6137692e-01  5.3105032e-01]
  [ 4.6053764e-01  5.4286075e-01  8.5167921e-01 ...  3.0370983e-01
    4.4904885e-01  4.9034703e-01]
  ...
  [ 5.3197891e-01  8.8493884e-01  2.2825547e-01 ...  8.8850749e-01
    7.6855040e-01  6.2100285e-01]
  [ 6.3237846e-01  5.2410227e-01  6.0207236e-01 ...  4.9969846e-01
    3.5816756e-01  2.0835558e-01]
  [-4.7841255e-02  5.2957022e-01  7.9817265e-02 ...  5.8414381e-02
    8.1727415e-02  5.8729213e-01]]

 [[ 7.6542306e-01  6.9060016e-01  3.3734241e-01 ...  6.4656264e-01
    2.7471843e-01  4.4688481e-01]
  [ 7.3970360e-01  9.8374236e-01  3.4647760e-01 ...  3.6925513e-01
    2.6154196e-01  2.3759985e-01]
  [-5.0112538e-02  4.6827608e-01  7.3408574e-01 ...  4.2065576e-01
    5.8828568e-01  2.3681982e-01]
  ...
  [ 5.7841629e-01  2.3107496e-01  5.9508163e-01 ... -4.7579780e-04
    6.9405937e-01  7.4537039e-01]
  [ 4.7800592e-01  7.3489493e-01  6.3554561e-01 ...  6.2975085e-01
    5.9424227e-01  3.4895000e-01]
  [ 6.1014187e-01  6.8143499e-01  4.3950918e-01 ...  7.4912435e-01
    8.7687171e-01  2.2635195e-01]]

 [[ 7.0224035e-01  6.9118708e-01  8.0853140e-01 ...  6.7778695e-01
    3.0537796e-01  2.7040458e-01]
  [ 1.8654233e-01  9.6552259e-01  4.7433805e-02 ...  8.9787990e-01
    3.9053717e-01  2.0925885e-01]
  [ 3.1007111e-01  7.5688696e-01  4.0302896e-01 ...  2.5267088e-01
    3.1179979e-01  2.0612539e-01]
  ...
  [ 5.6390470e-01  7.3319811e-01  1.1306577e-01 ...  6.4878613e-01
    6.4180237e-01  7.2481465e-01]
  [ 7.4073094e-01  6.5915877e-01  4.4098765e-02 ...  4.5381948e-01
    5.2133381e-01  5.3710729e-01]
  [ 5.0160265e-01  9.6063000e-01  5.7726550e-01 ...  8.1778222e-01
    7.1846193e-01  8.6681777e-01]]

 [[ 5.5969074e-02  3.4243438e-01  2.6192406e-01 ...  5.8297426e-01
    9.2344010e-01  9.7105581e-01]
  [ 3.0831653e-01  8.4483176e-01  6.6090798e-01 ...  2.4024777e-01
    3.5374004e-01  5.8804435e-01]
  [ 8.6001533e-01  9.6720868e-01  6.2938458e-01 ... -6.3361019e-02
    6.5138531e-01  1.3370466e-01]
  ...
  [ 8.0825627e-01  7.9345517e-02  3.2589832e-01 ...  5.2596655e-02
    8.7211215e-01  1.7172907e-01]
  [ 8.1158543e-01  7.9371500e-01  7.5868255e-01 ...  2.6026982e-01
    1.0206559e+00  9.3225348e-01]
  [ 3.9073047e-01  6.0695231e-01  7.4296314e-01 ...  2.4992853e-01
    2.0046885e-01  6.4431584e-01]]]


[[[ 4.7617313e-01  7.5242633e-01  5.9879410e-01 ...  5.0899339e-01
    2.6455650e-01  5.1510149e-01]
  [ 2.5535119e-01  6.8977267e-01  4.1920286e-01 ...  7.6484394e-01
    6.3488644e-01  6.4388931e-01]
  [ 4.1237402e-01  1.7340298e-01  6.7449671e-01 ...  2.1270150e-01
    6.3466746e-01  8.6857831e-01]
  ...
  [ 8.1065565e-02  6.6501483e-02  5.6959128e-01 ...  5.4045898e-01
    8.6184585e-01  9.3150401e-01]
  [-7.6004677e-02  5.4888248e-01  8.7145883e-01 ...  6.0898107e-01
    6.4421803e-01  8.7105870e-01]
  [ 4.6014377e-01  9.1836780e-01 -5.3355824e-03 ...  4.1441053e-01
    7.9786223e-01  8.9707732e-01]]

 [[ 5.6181866e-01  9.2686254e-01  3.0097459e-02 ... -3.8611926e-03
    4.7412282e-01  4.4197452e-01]
  [ 5.8265114e-01  7.3647016e-01  7.0062947e-01 ...  6.4639241e-01
    7.6137680e-01  5.3105026e-01]
  [ 4.6053758e-01  5.4286069e-01  8.5167921e-01 ...  3.0370986e-01
    4.4904888e-01  4.9034697e-01]
  ...
  [ 5.3197896e-01  8.8493884e-01  2.2825548e-01 ...  8.8850754e-01
    7.6855028e-01  6.2100285e-01]
  [ 6.3237852e-01  5.2410227e-01  6.0207236e-01 ...  4.9969843e-01
    3.5816756e-01  2.0835567e-01]
  [-4.7841284e-02  5.2957022e-01  7.9817221e-02 ...  5.8414415e-02
    8.1727415e-02  5.8729213e-01]]

 [[ 7.6542324e-01  6.9060022e-01  3.3734241e-01 ...  6.4656276e-01
    2.7471834e-01  4.4688484e-01]
  [ 7.3970360e-01  9.8374236e-01  3.4647760e-01 ...  3.6925519e-01
    2.6154196e-01  2.3759985e-01]
  [-5.0112501e-02  4.6827611e-01  7.3408568e-01 ...  4.2065573e-01
    5.8828568e-01  2.3681976e-01]
  ...
  [ 5.7841629e-01  2.3107493e-01  5.9508163e-01 ... -4.7583878e-04
    6.9405937e-01  7.4537039e-01]
  [ 4.7800595e-01  7.3489487e-01  6.3554561e-01 ...  6.2975079e-01
    5.9424227e-01  3.4895009e-01]
  [ 6.1014187e-01  6.8143505e-01  4.3950915e-01 ...  7.4912429e-01
    8.7687171e-01  2.2635195e-01]]

 [[ 7.0224029e-01  6.9118714e-01  8.0853134e-01 ...  6.7778695e-01
    3.0537799e-01  2.7040464e-01]
  [ 1.8654232e-01  9.6552259e-01  4.7433816e-02 ...  8.9787990e-01
    3.9053714e-01  2.0925885e-01]
  [ 3.1007102e-01  7.5688696e-01  4.0302888e-01 ...  2.5267094e-01
    3.1179985e-01  2.0612541e-01]
  ...
  [ 5.6390482e-01  7.3319811e-01  1.1306578e-01 ...  6.4878607e-01
    6.4180243e-01  7.2481471e-01]
  [ 7.4073100e-01  6.5915883e-01  4.4098806e-02 ...  4.5381945e-01
    5.2133375e-01  5.3710735e-01]
  [ 5.0160259e-01  9.6063006e-01  5.7726556e-01 ...  8.1778222e-01
    7.1846187e-01  8.6681777e-01]]

 [[ 5.5969067e-02  3.4243438e-01  2.6192403e-01 ...  5.8297431e-01
    9.2344004e-01  9.7105587e-01]
  [ 3.0831656e-01  8.4483165e-01  6.6090798e-01 ...  2.4024776e-01
    3.5374004e-01  5.8804429e-01]
  [ 8.6001533e-01  9.6720868e-01  6.2938464e-01 ... -6.3361034e-02
    6.5138531e-01  1.3370466e-01]
  ...
  [ 8.0825627e-01  7.9345502e-02  3.2589832e-01 ...  5.2596629e-02
    8.7211215e-01  1.7172906e-01]
  [ 8.1158543e-01  7.9371500e-01  7.5868255e-01 ...  2.6026982e-01
    1.0206559e+00  9.3225348e-01]
  [ 3.9073041e-01  6.0695231e-01  7.4296314e-01 ...  2.4992855e-01
    2.0046882e-01  6.4431584e-01]]]
    
    """
    ## 绝对误差小于 1e-5
    assert np.allclose(res_cat, fuse_res,atol=1e-5)

if __name__ == '__main__':
    
    # Two models test
    data = np.random.random((5, 20,4096)).astype(np.float32)
    test_res(2, [0,1,5], data)
    # for i in range(0,6):
    #     test_res(2, [0, i, 5], data)

    # # 5 models test
    # data = np.random.random((10,784)).astype(np.float32)
    # test_res(5, [0,3,5,7,8,10], data)
