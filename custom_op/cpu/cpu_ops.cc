// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_lite_custom_op.h"
#include <iostream>

#define CUSTOM_ENFORCE(cond, msg)                                \
  if (!(cond)) {                                                 \
    ORT_CXX_API_THROW(msg, OrtErrorCode::ORT_RUNTIME_EXCEPTION); \
  }

using namespace Ort::Custom;

namespace Cpu {

struct KernelOne {
  KernelOne(const OrtApi*, const OrtKernelInfo*) {}

  Ort::Status Compute(const Ort::Custom::Tensor<float>& X,
                      const Ort::Custom::Tensor<float>& Y,
                      Ort::Custom::Tensor<float>& Z) {
    if (X.NumberOfElement() != Y.NumberOfElement()) {
      return Ort::Status("x and y has different number of elements", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    auto x_shape = X.Shape();
    auto x_raw = X.Data();
    auto y_raw = Y.Data();
    auto z_raw = Z.Allocate(x_shape);
    for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
      z_raw[i] = x_raw[i] + y_raw[i];
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count != 2) {
      return Ort::Status("input count should be 2", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    Ort::ShapeInferContext::Shape shape_3_5 = {{3}, {5}};
    if (ctx.GetInputShape(0) != shape_3_5 ||
        ctx.GetInputShape(1) != shape_3_5) {
      return Ort::Status("input shape mismatch", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    return Ort::Status{nullptr};
  }
};

struct DataI {
  const float* from = {};
  float* to = {};
};

struct DataII {
  const float* from = {};
  int32_t* to = {};
};

// floats to floats
void CopyI(void* raw_data, size_t ith) {
  auto data = reinterpret_cast<DataI*>(raw_data);
  data->to[ith] = data->from[ith];
}

// floats to int32_t
void CopyII(void* raw_data, size_t ith) {
  auto data = reinterpret_cast<DataII*>(raw_data);
  data->to[ith] = static_cast<int32_t>(round(data->from[ith]));
}

// lite custom op as a function
void KernelTwo(OrtKernelContext* context,
               const Ort::Custom::Tensor<float>& X,
               Ort::Custom::Tensor<int32_t>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  std::vector<float> floats(static_cast<size_t>(X.NumberOfElement()), 0.f);

  DataI data_i = {X_raw, floats.data()};
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  Ort::KernelContext ctx(context);
  ctx.ParallelFor(CopyI, static_cast<size_t>(total), 0, &data_i);  // test simple parallel for

  DataII data_ii = {floats.data(), Y_raw};
  ctx.ParallelFor(CopyII, static_cast<size_t>(total), 2, &data_ii);  // test batch parallel for
}

template <typename T>
void MulTop(const Ort::Custom::Span<T>& in, Ort::Custom::Tensor<T>& out) {
  out.Allocate({1})[0] = in[0] * in[1];
}

void Fuse(
    OrtKernelContext*,
    const Ort::Custom::Span<float>& vector_1,
    const Ort::Custom::Span<float>& vector_2,
    int32_t alpha,
    Ort::Custom::Tensor<float>& vector_output) {
  auto len_output = std::min(vector_1.size(), vector_2.size());
  float* floats_out = static_cast<float*>(vector_output.Allocate({(int64_t)len_output}));
  for (size_t i = 0; i < len_output; ++i) {
    floats_out[i] = (vector_1[i] + vector_2[i]) * alpha;
  }
}

void Select(const Ort::Custom::Span<int32_t>& indices_in,
            Ort::Custom::Tensor<int32_t>& indices_out) {
  std::vector<int32_t> selected_indices;
  for (size_t i = 0; i < indices_in.size(); ++i) {
    if (indices_in[i] % 2 == 0) {
      selected_indices.push_back(indices_in[i]);
    }
  }

  int32_t* int_out = static_cast<int32_t*>(indices_out.Allocate({static_cast<int64_t>(selected_indices.size())}));
  for (size_t j = 0; j < selected_indices.size(); ++j) {
    int_out[j] = selected_indices[j];
  }
}

struct Filter {
  Filter(const OrtApi*, const OrtKernelInfo*) {}
  Ort::Status Compute(const Ort::Custom::Tensor<float>& floats_in,
                      Ort::Custom::Tensor<float>& floats_out) {
    const float* in = floats_in.Data();
    auto in_len = floats_in.NumberOfElement();

    std::vector<float> filter_floats;
    for (int64_t i = 0; i < in_len; ++i) {
      if (in[i] > 1.f) {
        filter_floats.push_back(in[i]);
      }
    }

    float* out = static_cast<float*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
    for (size_t j = 0; j < filter_floats.size(); ++j) {
      out[j] = filter_floats[j];
    }

    return Ort::Status{nullptr};
  }
};

void Box(const Ort::Custom::Tensor<float>* float_in_1,
         const Ort::Custom::Tensor<float>* float_in_2,
         std::optional<const Ort::Custom::Tensor<float>*> float_in_3,
         Ort::Custom::Tensor<float>* float_out_1,
         std::optional<Ort::Custom::Tensor<float>*> float_out_2) {
  auto raw_in_1 = float_in_1->Data();
  auto raw_in_2 = float_in_2->Data();

  auto l_in_1 = float_in_1->Shape()[0];
  auto l_in_2 = float_in_2->Shape()[0];
  auto l_out_1 = l_in_1 + l_in_2;

  auto raw_out_1 = float_out_1->Allocate({l_out_1});

  for (int64_t i = 0; i < l_out_1; ++i) {
    raw_out_1[i] = i < l_in_1 ? raw_in_1[i] : raw_in_2[i - l_in_1];
  }

  if (float_in_3.has_value() && float_out_2.has_value()) {
    auto raw_in_3 = float_in_3.value()->Data();
    auto l_in_3 = float_in_3.value()->Shape()[0];
    auto l_out_2 = l_in_2 + l_in_3;
    auto raw_out_2 = float_out_2.value()->Allocate({l_out_2});
    for (int64_t i = 0; i < l_out_2; ++i) {
      raw_out_2[i] = i < l_in_2 ? raw_in_2[i] : raw_in_3[i - l_in_2];
    }
  }
}

Ort::Status Merge(const Ort::Custom::TensorArray& inputs, Ort::Custom::TensorArray& outputs) {

    auto output_shape = inputs[0]->Shape();
    output_shape[0] = 0;
    for (size_t ith_input = 0; ith_input < inputs.Size(); ++ith_input) {
        output_shape[0] += inputs[ith_input]->Shape()[0];
    }
    float* raw_output_0 = outputs.AllocateOutput<float>(0, output_shape);
    
    // 分配索引tensor
    int32_t* raw_output_1 = outputs.AllocateOutput<int32_t>(1, {inputs.Size() + 1});
    size_t index = 0;
    size_t start = 0;
    raw_output_1[0] = 0; // 第一个索引总是从0开始

    // 按顺序复制每个输入tensor的数据到输出tensor
    for (size_t ith_input = 0; ith_input < inputs.Size(); ++ith_input) {
        const auto& input = inputs[ith_input];
        const float* raw_input = reinterpret_cast<const float*>(input->DataRaw());
        size_t num_elements = input->NumberOfElement();

        // 更新索引tensor
        index += input->Shape()[0];
        raw_output_1[ith_input + 1] = static_cast<int32_t>(index);

        // 复制数据
        memcpy(raw_output_0 + start, raw_input, num_elements * sizeof(float));
        start += num_elements;
    }

    return Ort::Status{nullptr};
}


Ort::Status Route(const Ort::Custom::Tensor<float>& data_tensor,
                  const Ort::Custom::Tensor<int32_t>& index_tensor,
                  Ort::Custom::TensorArray& outputs) {
    const float* data_raw = data_tensor.Data();
    const int32_t* index_raw = index_tensor.Data();
    int64_t num_outputs = index_tensor.NumberOfElement() - 1;
    int64_t scale = data_tensor.NumberOfElement() / data_tensor.Shape()[0];

    for (int64_t i = 0; i < num_outputs; ++i) {
        int64_t start_index = index_raw[i];
        int64_t end_index = index_raw[i + 1];
        int64_t num_elements = (end_index - start_index) * scale;

        // Assume that all tensors have the same shape except for the batch dimension
        std::vector<int64_t> output_shape = data_tensor.Shape();
        output_shape[0] = end_index - start_index;

        float* raw_output = outputs.AllocateOutput<float>(i, output_shape);
        memcpy(raw_output, data_raw + start_index * scale, num_elements * sizeof(float));
    }

    return Ort::Status{nullptr};
}


#if !defined(DISABLE_FLOAT8_TYPES)
struct KernelOneFloat8 {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const Ort::Float8E4M3FN_t* X = input_X.GetTensorData<Ort::Float8E4M3FN_t>();
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
    auto output = ctx.GetOutput(0, dimensions);
    Ort::Float8E4M3FN_t* out = output.GetTensorMutableData<Ort::Float8E4M3FN_t>();
    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i];
    }
  }
};
// legacy custom op registration
struct CustomOpOneFloat8 : Ort::CustomOpBase<CustomOpOneFloat8, KernelOneFloat8> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelOneFloat8>().release();
  };
  const char* GetName() const { return "CustomOpOneFloat8"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };
  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
};

void FilterFloat8(const Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_in,
                  Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_out) {
  const Ort::Float8E4M3FN_t* in = floats_in.Data();
  auto in_len = floats_in.NumberOfElement();

  std::vector<Ort::Float8E4M3FN_t> filter_floats;
  for (int64_t i = 0; i < in_len; ++i) {
    if (in[i] > 1.f) {
      filter_floats.push_back(in[i]);
    }
  }

  Ort::Float8E4M3FN_t* out = static_cast<Ort::Float8E4M3FN_t*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
  for (size_t j = 0; j < filter_floats.size(); ++j) {
    out[j] = filter_floats[j];
  }
}
#endif

// a sample custom op accepting variadic inputs, and generate variadic outputs by simply 1:1 copying.
template <typename T>
Ort::Status CopyTensorArrayAllVariadic(const Ort::Custom::TensorArray& inputs, Ort::Custom::TensorArray& outputs) {
  for (size_t ith_input = 0; ith_input < inputs.Size(); ++ith_input) {
    const auto& input = inputs[ith_input];
    const auto& input_shape = input->Shape();
    const T* raw_input = reinterpret_cast<const T*>(input->DataRaw());
    auto num_elements = input->NumberOfElement();
    T* raw_output = outputs.AllocateOutput<T>(ith_input, input_shape);
    if (!raw_output) {
      return Ort::Status("Failed to allocate output!", OrtErrorCode::ORT_FAIL);
    }
    for (int64_t jth_elem = 0; jth_elem < num_elements; ++jth_elem) {
      raw_output[jth_elem] = raw_input[jth_elem];
    }
  }
  return Ort::Status{nullptr};
}

template <typename T>
Ort::Status CopyTensorArrayCombined(const Ort::Custom::Tensor<float>& first_input,
                                    const Ort::Custom::TensorArray& other_inputs,
                                    Ort::Custom::Tensor<float>& first_output,
                                    Ort::Custom::TensorArray& other_outputs) {
  const auto first_input_shape = first_input.Shape();
  const T* raw_input = reinterpret_cast<const T*>(first_input.DataRaw());

  T* raw_output = first_output.Allocate(first_input_shape);
  if (!raw_output) {
    return Ort::Status("Failed to allocate output!", OrtErrorCode::ORT_FAIL);
  }

  auto num_elements = first_input.NumberOfElement();
  for (int64_t ith_elem = 0; ith_elem < num_elements; ++ith_elem) {
    raw_output[ith_elem] = raw_input[ith_elem];
  }

  return CopyTensorArrayAllVariadic<T>(other_inputs, other_outputs);
}

Ort::Status AttrTesterIntFloatCompute(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Z) {
  auto x_shape = X.Shape();
  auto x_raw = X.Data();
  auto z_raw = Z.Allocate(x_shape);
  for (int64_t i = 0; i < X.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] * 2;
  }
  return Ort::Status{nullptr};
}

Ort::Status AttrTesterIntFloatShapeInfer(Ort::ShapeInferContext& ctx) {
  CUSTOM_ENFORCE(ctx.GetAttrInt("a_int") == 1, "int attr mismatch");
  CUSTOM_ENFORCE(ctx.GetAttrFloat("a_float") == 2.f, "float attr mismatch");
  std::vector<int64_t> ints{3, 4, 5};
  CUSTOM_ENFORCE(ctx.GetAttrInts("ints") == ints, "ints attr mismatch");
  std::vector<float> floats{6, 7, 8};
  CUSTOM_ENFORCE(ctx.GetAttrFloats("floats") == floats, "floats attr mismatch");
  auto input_shape = ctx.GetInputShape(0);
  CUSTOM_ENFORCE(input_shape.size() == 1 &&
                     !input_shape[0].IsInt() &&
                     std::string{input_shape[0].AsSym()} == "d",
                 "input dim is not symbolic");
  ctx.SetOutputShape(0, input_shape);
  return Ort::Status{nullptr};
}

struct AttrTesterStringKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const auto* X = input_X.GetTensorData<float>();
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
    auto output = ctx.GetOutput(0, dimensions);
    auto* out = output.GetTensorMutableData<float>();
    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] * 3;
    }
  }
};

struct AttrTesterStringOp : Ort::CustomOpBase<AttrTesterStringOp, AttrTesterStringKernel> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const {
    return std::make_unique<AttrTesterStringKernel>().release();
  };
  const char* GetName() const { return "AttrTesterString"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };
  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    CUSTOM_ENFORCE(ctx.GetAttrString("a_string") == "iamastring", "string attr mismatch");
    std::vector<std::string> strings{"more", "strings"};
    CUSTOM_ENFORCE(ctx.GetAttrStrings("strings") == strings, "strings attr mismatch");
    return Ort::Status{nullptr};
  }
};

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp<KernelOne>("CustomOpOne", "CPUExecutionProvider")};
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", "CPUExecutionProvider", KernelTwo)};
  static const std::unique_ptr<OrtLiteCustomOp> c_MulTopOpFloat{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<float>)};
  static const std::unique_ptr<OrtLiteCustomOp> c_MulTopOpInt32{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<int32_t>)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Fuse{Ort::Custom::CreateLiteCustomOp("Fuse", "CPUExecutionProvider", Fuse, {}, 10, 12)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Select{Ort::Custom::CreateLiteCustomOp("Select", "CPUExecutionProvider", Select)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Filter{Ort::Custom::CreateLiteCustomOp<Filter>("Filter", "CPUExecutionProvider", 15, 17)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Box{Ort::Custom::CreateLiteCustomOp("Box", "CPUExecutionProvider", Box)};

  static const std::unique_ptr<OrtLiteCustomOp> c_Merge{Ort::Custom::CreateLiteCustomOp("Merge", "CPUExecutionProvider", Merge)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Route{Ort::Custom::CreateLiteCustomOp("Route", "CPUExecutionProvider", Route)};
  static const std::unique_ptr<OrtLiteCustomOp> c_CopyTensorArrayAllVariadic{Ort::Custom::CreateLiteCustomOp("CopyTensorArrayAllVariadic", "CPUExecutionProvider", CopyTensorArrayAllVariadic<float>)};
  static const std::unique_ptr<OrtLiteCustomOp> c_CopyTensorArrayCombined{Ort::Custom::CreateLiteCustomOp("CopyTensorArrayCombined", "CPUExecutionProvider", CopyTensorArrayCombined<float>)};

  static const std::unique_ptr<OrtLiteCustomOp> c_AtterTesterIntFloat{Ort::Custom::CreateLiteCustomOp("AttrTesterIntFloat", "CPUExecutionProvider", AttrTesterIntFloatCompute, AttrTesterIntFloatShapeInfer)};
  static const AttrTesterStringOp c_AtterTesterString;

#if !defined(DISABLE_FLOAT8_TYPES)
  static const CustomOpOneFloat8 c_CustomOpOneFloat8;
  static const std::unique_ptr<OrtLiteCustomOp> c_FilterFloat8{Ort::Custom::CreateLiteCustomOp("FilterFloat8", "CPUExecutionProvider", FilterFloat8)};
#endif

  domain.Add(c_CustomOpOne.get());
  domain.Add(c_CustomOpTwo.get());
  domain.Add(c_MulTopOpFloat.get());
  domain.Add(c_MulTopOpInt32.get());
  domain.Add(c_Fuse.get());
  domain.Add(c_Select.get());
  domain.Add(c_Filter.get());
  domain.Add(c_Box.get());
  domain.Add(c_CopyTensorArrayAllVariadic.get());
  domain.Add(c_CopyTensorArrayCombined.get());
  domain.Add(c_AtterTesterIntFloat.get());
  domain.Add(c_Merge.get());
  domain.Add(c_Route.get());
  domain.Add(&c_AtterTesterString);

#if !defined(DISABLE_FLOAT8_TYPES)
  domain.Add(&c_CustomOpOneFloat8);
  domain.Add(c_FilterFloat8.get());
#endif
}

}  // namespace Cpu
