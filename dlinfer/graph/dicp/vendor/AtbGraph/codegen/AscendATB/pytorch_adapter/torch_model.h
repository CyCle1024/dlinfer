#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "model.h"

class TorchModel : public torch::CustomClassHolder {
 public:
  TorchModel(std::string modelName);
  ~TorchModel();
  int64_t SetParam(std::string param);
  std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors,
                                     std::string param);
  int64_t ExecuteOut(std::vector<torch::Tensor> atInTensors,
                     std::vector<torch::Tensor> atOutTensors,
                     std::string param);
  c10::intrusive_ptr<TorchModel> clone() const {
    return c10::make_intrusive<TorchModel>(modelName_);
  }

 private:
  int64_t TransferAtTensor2AtbTensor(std::vector<torch::Tensor>& atTensors,
                          std::vector<atb::Tensor>& opsTensors);
  int64_t ExecuteOutImpl(std::vector<atb::Tensor>& inTensors,
                         std::vector<atb::Tensor>& outTensors,
                         const std::string& param);
  std::string GetSaveTensorDir();
  void* GetWorkSpace(uint64_t bufferSize);
  atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc& tensorDesc);
  void RunTask(std::string taskName, std::function<int()> task);

 private:
  std::string modelName_;
  std::shared_ptr<dicp::Model> model_;
  uint64_t executeCount_ = 0;
  uint64_t modelId_ = 0;
  std::shared_ptr<atb::Context> context_;
  std::vector<torch::Tensor> atInternalTensors_;
};
