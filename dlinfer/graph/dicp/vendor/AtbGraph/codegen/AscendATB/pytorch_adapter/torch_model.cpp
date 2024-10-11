

#include "torch_model.h"

#include <acl/acl.h>
#include <atb/utils.h>
#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "config.h"
#include "context_factory.h"
#include "dicp_model.h"
#include "log.h"
#include "tensor_utils.h"
#include "utils.h"
#include "workspace.h"

using namespace dicp;

void* TorchModel::GetWorkSpace(uint64_t bufferSize) {
  void* workspace = nullptr;
  if (bufferSize > 0) {
    workspace = GetWorkspaceBuffer(bufferSize);
  }
  return workspace;
}

atb::Tensor TorchModel::CreateInternalTensorFromDesc(
    const atb::TensorDesc& tensorDesc) {
  torch::Tensor newAtTensor = tensor_utils::CreateAtTensorFromTensorDesc(tensorDesc);
  atInternalTensors_.push_back(newAtTensor);
  return tensor_utils::AtTensor2Tensor(newAtTensor);
}

void TorchModel::RunTask(std::string taskName, std::function<int()> task) {
#ifdef TORCH_SETCUSTOMHANDLER
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
#else
  DICP_LOG(FATAL) << "torch_npu is low, can't support SetCustomHandler";
#endif
}

uint64_t GetNewModelId() {
  static uint64_t modelId = 0;
  uint64_t newModelId = modelId++;
  return newModelId;
}

TorchModel::TorchModel(std::string modelName) : modelName_(modelName) {
  modelId_ = GetNewModelId();
  context_ = ContextFactory::GetAtbContext(utils::GetCurrentStream());
  DICP_LOG(INFO) << "TorchModel new modelId:" << modelId_;
}

TorchModel::~TorchModel() {
  context_.reset();
  ContextFactory::FreeAtbContext();
};

int64_t TorchModel::SetParam(std::string param) {
  DICP_LOG(INFO) << "TorchModel set param start, modelId:" << modelId_
                 << ", param:" << param;

  model_ = std::make_shared<DICPModel>(param);
  const char* taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
  const char* blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
  bool isTaskQueueEnable = GetConfig().IsTaskQueueEnable();

  auto getWorkspaceFunc =
      std::bind(&TorchModel::GetWorkSpace, this, std::placeholders::_1);
  auto createInternalTensorFromDescFunc = std::bind(
      &TorchModel::CreateInternalTensorFromDesc, this, std::placeholders::_1);
  auto runTaskFunc = std::bind(&TorchModel::RunTask, this,
                               std::placeholders::_1, std::placeholders::_2);
  int64_t atbStatus = 0;
  if (isTaskQueueEnable) {
    atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc,
                             runTaskFunc);
  } else {
    atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc,
                             nullptr);
  }
  DICP_LOG(INFO) << "TorchModel set param end";
  return atbStatus;
}

std::vector<torch::Tensor> TorchModel::Execute(
    std::vector<torch::Tensor> atInTensors, std::string param) {
  atInternalTensors_.clear();
  for (size_t i = 0; i < atInTensors.size(); ++i) {
    const torch::Tensor& atTensor = atInTensors.at(i);
    DICP_LOG(INFO) << "TorchModel atInTensors[" << i << "]"
                   << " data:" << atTensor.data_ptr()
                   << ", storage_offset:" << atTensor.storage_offset()
                   << ", format:" << tensor_utils::GetTensorNpuFormat(atTensor)
                   << ", shape:" << atTensor.sizes()
                   << ", options:" << atTensor.options();
  }

  std::vector<atb::Tensor> inTensors;
  TransferAtTensor2AtbTensor(atInTensors, inTensors);
  if (GetConfig().IsConvertNCHWToND()) {
    for (auto& inTensor : inTensors) {
      if (inTensor.desc.format == ACL_FORMAT_NCHW) {
        inTensor.desc.format = ACL_FORMAT_ND;
      }
    }
  }
  std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
  for (size_t i = 0; i < inTensors.size(); ++i) {
    inTensorDescs.at(i) = inTensors.at(i).desc;
  }
  std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
  atb::Status st = model_->PreProcess(param);
  DICP_LOG_IF(st != 0, FATAL)
      << "TorchModel PreProcess fail, error code: " << st;
  st = model_->InferShape(inTensorDescs, outTensorDescs);
  DICP_LOG_IF(st != 0, FATAL)
      << "TorchModel infer shape fail, error code: " << st;

  std::vector<torch::Tensor> atOutTensors(outTensorDescs.size());
  for (size_t i = 0; i < atOutTensors.size(); ++i) {
    DICP_LOG(INFO) << "TorchModel outTensorDescs[" << i << "]:"
                   << tensor_utils::TensorDescToString(
                          outTensorDescs.at(i));
    atOutTensors.at(i) =
        tensor_utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
  }

  std::vector<atb::Tensor> outTensors;
  TransferAtTensor2AtbTensor(atOutTensors, outTensors);
  if (GetConfig().IsConvertNCHWToND()) {
    for (auto& outTensor : outTensors) {
      if (outTensor.desc.format == ACL_FORMAT_NCHW) {
        outTensor.desc.format = ACL_FORMAT_ND;
      }
    }
  }

  int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
  if (atbStatus != atb::NO_ERROR) {
    std::vector<torch::Tensor> atNullOutTensors(0);
    return atNullOutTensors;
  }
  return atOutTensors;
}

int64_t TorchModel::ExecuteOut(std::vector<torch::Tensor> atInTensors,
                               std::vector<torch::Tensor> atOutTensors,
                               std::string param) {
  atInternalTensors_.clear();
  std::vector<atb::Tensor> inTensors;
  TransferAtTensor2AtbTensor(atInTensors, inTensors);

  std::vector<atb::Tensor> outTensors;
  TransferAtTensor2AtbTensor(atOutTensors, outTensors);

  int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
  return atbStatus;
}

int64_t TorchModel::ExecuteOutImpl(std::vector<atb::Tensor>& inTensors,
                                   std::vector<atb::Tensor>& outTensors,
                                   const std::string& param) {
  int64_t atbStatus =
      model_->Execute(context_.get(), inTensors, outTensors, param);
  executeCount_++;
  return atbStatus;
}

int64_t TorchModel::TransferAtTensor2AtbTensor(
    std::vector<torch::Tensor>& atTensors,
    std::vector<atb::Tensor>& opsTensors) {
  for (auto& atTensor : atTensors) {
    tensor_utils::ContiguousAtTensor(atTensor);
    atb::Tensor tensor = tensor_utils::AtTensor2Tensor(atTensor);
    opsTensors.push_back(tensor);
  }
  return atb::NO_ERROR;
}

TORCH_LIBRARY(TorchModel, m) {
  m.class_<TorchModel>("TorchModel")
      .def(torch::init<std::string>())
      .def("set_param", &TorchModel::SetParam)
      .def("execute", &TorchModel::Execute)
      .def("execute_out", &TorchModel::ExecuteOut);
}
