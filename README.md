# Assumptions

For an operator to be eligible for fusion, it must meet the following conditions:

1. It has only one input, excluding `Constant` and `initializer` type tensors.
2. It has only one output.
3. The first dimension of both input and output shapes is annotated with "batch_size".

Therefore, we must first perform a more accurate shape inference, i.e., `symbolic shape infer`. Run the following command:

```bash
python ./tools/symbolic_shape_infer.py --input [input model path] --output [output model path]
```

# Usage

1. Download the onnxruntime project from https://github.com/microsoft/onnxruntime and build it from source by executing the following commands:

   ```bash
   git clone https://github.com/microsoft/onnxruntime.git
   cd onnxruntime
   git apply ./runtime/ort/changes.patches
   ```

2. Install the Python package:

   ```bash
   pip install -e .
   ```

# Examples

We have currently implemented custom CPU ops [Merge and Route] for onnxruntime.

## Microbenchmark

In the `./example/micro` directory, you can find some files. Follow these instructions to test the functionality for microbenchmark:

```bash
cd example/micro
python generate.py
./convert.sh

python fuse.py --num 2
python fuse.py
python test_runtime.py
```

## Transformer Example

In the `./example/transformer` directory, follow these instructions to test the functionality. We use two decode layers of the LLaMA model and its LoRA variant as our test models:

```bash
cd example/transformer
python generate.py
./convert.sh

python fuse.py
python test_runtime.py
```

<!-- # Additional Tools

We provide a script `./tools/constant_to_initializer.py` that converts `Constant` nodes into initializers. You can use this script to create a graph without `Constant` nodes, which is necessary for our fusion algorithm. -->

# TODO
- [ ] Generalize input assumptions to handle multiple inputs
- [ ] Refactor the single Route Op into multiple specialized Route Ops.
