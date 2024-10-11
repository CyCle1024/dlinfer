#pragma once

#include <atb/types.h>
#include <torch/torch.h>

#include <vector>

namespace dicp {
namespace tensor_utils {

std::string TensorToString(const atb::Tensor& tensor);
std::string TensorDescToString(const atb::TensorDesc& tensorDesc);
uint64_t GetTensorNumel(const atb::Tensor& tensor);
uint64_t GetTensorNumel(const atb::TensorDesc& tensorDesc);
bool TensorDescEqual(const atb::TensorDesc& tensorDescA,
                     const atb::TensorDesc& tensorDescB);

atb::Tensor AtTensor2Tensor(const at::Tensor& atTensor);
at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc& tensorDesc);
at::Tensor NpuFormatCast(const at::Tensor& tensor);
bool AtTensorShapeEqualToTensor(const at::Tensor& atTensor,
                                const atb::TensorDesc& tensorDesc);
int64_t GetTensorNpuFormat(const at::Tensor& tensor);
void ContiguousAtTensors(std::vector<torch::Tensor>& atTensors);
void ContiguousAtTensor(torch::Tensor& atTensor);

} // namespace tensor_utils
} // namespace dicp