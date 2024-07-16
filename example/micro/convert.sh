for i in $(seq 1 5); do
    python ../../tools/symbolic_shape_infer.py --input ./org_model/model_$i.onnx --output ./model/model_$i.onnx
    # python ../../tools/constant_to_initializer.py --input ./model/model_$i.onnx --output ./model/model_$i.onnx
done