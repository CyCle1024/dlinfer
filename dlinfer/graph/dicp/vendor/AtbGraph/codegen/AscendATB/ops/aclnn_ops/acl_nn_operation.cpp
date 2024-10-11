#include "acl_nn_operation.h"

#include "log.h"

namespace dicp {

int AclNnTensor::CreateTensor(const std::string& opName) {
  atb::SVector<int64_t> tmpStrides(atbTensor.desc.shape.dimNum, 1);
  for (int64_t i = atbTensor.desc.shape.dimNum - 2; i >= 0; i--) {
    tmpStrides[i] = atbTensor.desc.shape.dims[i + 1] * tmpStrides[i + 1];
  }
  strides = tmpStrides;

  DICP_LOG(INFO) << opName << " aclCreateTensor start, tensor.deviceData:"
                 << atbTensor.deviceData;
  tensor =
      aclCreateTensor(atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
                      atbTensor.desc.dtype, strides.data(), 0,
                      atbTensor.desc.format, atbTensor.desc.shape.dims,
                      atbTensor.desc.shape.dimNum, atbTensor.deviceData);
  if (tensor) {
    DICP_LOG(INFO) << opName << " aclCreateTensor success, tensor:" << tensor;
    return atb::NO_ERROR;
  }

  DICP_LOG(ERROR) << opName << " aclCreateTensor fail";
  return atb::ERROR_INTERNAL_ERROR;
}

int AclNnTensor::CreateTransposeTensor(const std::string& opName) {
  atb::SVector<int64_t> tmptransposeStrides(atbTensor.desc.shape.dimNum, 1);
  tmptransposeStrides[0] = 1;
  tmptransposeStrides[1] = atbTensor.desc.shape.dims[0];
  strides = tmptransposeStrides;

  DICP_LOG(INFO) << opName
                 << " aclCreateTransposeTensor start, tensor.deviceData:"
                 << atbTensor.deviceData;
  tensor =
      aclCreateTensor(atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
                      atbTensor.desc.dtype, strides.data(), 0,
                      atbTensor.desc.format, atbTensor.desc.shape.dims,
                      atbTensor.desc.shape.dimNum, atbTensor.deviceData);
  if (tensor) {
    DICP_LOG(INFO) << opName
                   << " aclCreateTransposeTensor success, tensor:" << tensor;
    return atb::NO_ERROR;
  }

  DICP_LOG(ERROR) << opName << " aclCreateTransposeTensor fail";
  return atb::ERROR_INTERNAL_ERROR;
}

int AclNnTensor::InitTensor(void* executor, const std::string& opName,
                            const size_t index, bool isInput) {
  if (!tensor) {
    DICP_LOG(ERROR) << opName << " acl tensor is null, not call aclInitTensor";
    return atb::ERROR_INTERNAL_ERROR;
  }

  DICP_LOG(INFO) << opName << " aclInitTensor start, tensor:" << tensor
                 << ", tensor.deviceData:" << atbTensor.deviceData;

  int ret = 0;
  if (isInput) {
    ret = AclSetInputTensorAddr(static_cast<aclOpExecutor*>(executor), index,
                                tensor, atbTensor.deviceData);
  } else {
    ret = AclSetOutputTensorAddr(static_cast<aclOpExecutor*>(executor), index,
                                 tensor, atbTensor.deviceData);
  }

  DICP_LOG_IF(ret != 0, ERROR)
      << opName << " aclInitTensor fail, error:" << ret;
  return ret;
}

void AclNnTask::Destory() {
  for (size_t i = 0; i < aclInTensors.size(); ++i) {
    aclDestroyTensor(aclInTensors[i].tensor);
  }
  aclInTensors.clear();

  for (size_t i = 0; i < aclOutTensors.size(); ++i) {
    aclDestroyTensor(aclOutTensors[i].tensor);
  }
  aclOutTensors.clear();

  for (size_t i = 0; i < aclInIntArrays.size(); ++i) {
    aclDestroyIntArray(aclInIntArrays[i]);
  }
  aclInIntArrays.clear();
}

AclNnOperation::AclNnOperation(const std::string& opName) : opName_(opName) {}

AclNnOperation::~AclNnOperation() {}

std::string AclNnOperation::GetName() const { return opName_; }

static const uint64_t ACTIVATION_INDEX = 0;
static const uint64_t BIAS_INDEX = 4;

atb::Status AclNnOperation::Setup(const atb::VariantPack& variantPack,
                                  uint64_t& workspaceSize,
                                  atb::Context* context) {
  DICP_LOG(INFO) << opName_ << " setup start";

  if (context == nullptr) {
    DICP_LOG(ERROR) << opName_ << " setup context is null";
    return atb::ERROR_INVALID_PARAM;
  }

  int ret = CreateAclTensors(variantPack, aclNnTask_);
  if (ret != 0) {
    DICP_LOG(ERROR) << opName_ << " call acl create tensor fail, error:" << ret;
    return atb::ERROR_CANN_ERROR;
  }
  for (size_t i = 0; i < aclNnTask_.aclInTensors.size(); ++i) {
    int ret = 0;
    ret = aclNnTask_.aclInTensors.at(i).CreateTensor(opName_);
    if (ret != 0) {
      return atb::ERROR_INTERNAL_ERROR;
    }
  }

  for (size_t i = 0; i < aclNnTask_.aclOutTensors.size(); ++i) {
    int ret = aclNnTask_.aclOutTensors.at(i).CreateTensor(opName_);
    if (ret != 0) {
      return atb::ERROR_INTERNAL_ERROR;
    }
  }

  ret = CallAclGetWorkspace(aclNnTask_, workspaceSize);
  if (ret != 0) {
    DICP_LOG(ERROR) << opName_ << " call acl get workspace fail, error:" << ret;
    return atb::ERROR_CANN_ERROR;
  }
  return atb::NO_ERROR;
}

atb::Status AclNnOperation::Execute(const atb::VariantPack& variantPack,
                                    uint8_t* workspace, uint64_t workspaceSize,
                                    atb::Context* context) {
  DICP_LOG(INFO) << opName_ << " execute start";
  if (!context) {
    DICP_LOG(ERROR) << opName_ << " execute fail, context param is null";
    return atb::ERROR_INVALID_PARAM;
  }

  aclrtStream stream = context->GetExecuteStream();
  if (!stream) {
    DICP_LOG(ERROR) << opName_
                    << " execute fail, execute stream in context is null";
    return atb::ERROR_INVALID_PARAM;
  }

  // 更新数据传入的地址
  int ret = UpdateAclTensorDataPtr(variantPack);
  if (ret != 0) {
    DICP_LOG(ERROR) << opName_ << " call acl init tensor fail, error:" << ret;
    aclNnTask_.Destory();
    return atb::ERROR_CANN_ERROR;
  }

  ret =
      CallAclExecute(workspace, workspaceSize, aclNnTask_.aclExecutor, stream);
  if (ret != 0) {
    DICP_LOG(ERROR) << opName_ << " call acl execute fail, error:" << ret;
    aclNnTask_.Destory();
    return atb::ERROR_CANN_ERROR;
  }

  aclNnTask_.Destory();

  DICP_LOG(INFO) << opName_ << " execute end";

  return atb::NO_ERROR;
}

atb::Status AclNnOperation::UpdateAclTensorDataPtr(
    const atb::VariantPack& variantPack) {
  for (size_t i = 0; i < aclNnTask_.aclInTensors.size(); ++i) {
    AclNnTensor& aclNnTensor = aclNnTask_.aclInTensors[i];
    if (aclNnTensor.needUpdateTensorDataPtr) {
      aclNnTensor.atbTensor.deviceData = variantPack.inTensors.at(i).deviceData;
      int ret =
          aclNnTensor.InitTensor(aclNnTask_.aclExecutor, opName_, i, true);
      if (ret != 0) {
        DICP_LOG(ERROR) << opName_ << " call InitTensor fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
      }
    }
  }

  for (size_t i = 0; i < aclNnTask_.aclOutTensors.size(); ++i) {
    AclNnTensor& aclNnTensor = aclNnTask_.aclOutTensors[i];
    if (aclNnTensor.needUpdateTensorDataPtr) {
      aclNnTensor.atbTensor.deviceData =
          variantPack.outTensors.at(i).deviceData;
      int ret =
          aclNnTensor.InitTensor(aclNnTask_.aclExecutor, opName_, i, false);
      if (ret != 0) {
        DICP_LOG(ERROR) << opName_ << " call InitTensor fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
      }
    }
  }

  return atb::NO_ERROR;
}
}  // namespace dicp