for i in $(seq 1 5); do
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input ./org_model/model_$i.onnx --output ./model/model_$i.onnx
done