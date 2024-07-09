# Usage
1. Download onnxruntime project from https://github.com/microsoft/onnxruntime, execute the following command and build onnxruntime from source.

```bash
git apply ./runtime/ort/changes.patches
```

2. Install python package

```bash
pip install -e .
```

# Example
Currently implement custom CPU ops [Merge and Route] for onnxruntime.

In the directory `./exmple/micro`, you can find some files. You can follow the instruction below to test the functionality for microbenchmark.

```bash
python generate.py

python fuse.py --num 2 # num: the number of fused model
python fuse.py --num 5

python test_runtime.py
```

