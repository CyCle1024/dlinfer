#include "add_rms_norm_operation.h"

#include <securec.h>
#include <syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm.h"
#include "log.h"

namespace dicp {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

AclNnAddRmsNormOperation::AclNnAddRmsNormOperation(const std::string& name,
                                                   float epsilon)
    : AclNnOperation(name) {
  this->epsilon = epsilon;
}

AclNnAddRmsNormOperation::~AclNnAddRmsNormOperation() {}

atb::Status AclNnAddRmsNormOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  DICP_LOG(INFO) << opName_ << " infer shape start";
  for (size_t i = 0; i < outTensorDescs.size(); i++) {
    outTensorDescs.at(i).format = inTensorDescs.at(0).format;
    if (i == NUM1) {
      outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
    } else {
      outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
    }

    outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
      DICP_LOG(INFO)
          << "[input0 dimNum = 3] CHECK W8A16_OP inputs shape: [input0]"
          << inTensorDescs.at(0).shape.dims[DIM0] << ", "
          << inTensorDescs.at(0).shape.dims[DIM1] << ", "
          << inTensorDescs.at(0).shape.dims[DIM2];
      outTensorDescs.at(i).shape.dims[DIM0] =
          inTensorDescs.at(0).shape.dims[DIM0];
      outTensorDescs.at(i).shape.dims[DIM1] =
          inTensorDescs.at(0).shape.dims[DIM1];
      outTensorDescs.at(i).shape.dims[DIM2] =
          inTensorDescs.at(0).shape.dims[DIM2];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
      DICP_LOG(INFO)
          << "[input0 dimNum = 2] CHECK W8A16_OP inputs shape: [input0]"
          << inTensorDescs.at(0).shape.dims[DIM0] << ", "
          << inTensorDescs.at(0).shape.dims[DIM1];
      outTensorDescs.at(i).shape.dims[DIM0] =
          inTensorDescs.at(0).shape.dims[DIM0];
      outTensorDescs.at(i).shape.dims[DIM1] =
          inTensorDescs.at(0).shape.dims[DIM1];
    } else {
      DICP_LOG(ERROR) << opName_ << " invalid dim num:"
                      << inTensorDescs.at(DIM0).shape.dimNum;
    }
  }

  DICP_LOG(INFO) << opName_ << " infer shape end";
  return 0;
}

uint32_t AclNnAddRmsNormOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnAddRmsNormOperation::GetOutputNum() const { return NUM3; }

int AclNnAddRmsNormOperation::CreateAclTensors(
    const atb::VariantPack& variantPack, AclNnTask& task) {
  DICP_LOG(INFO) << opName_ << " CreateAclTensor start";
  task.aclInTensors.resize(variantPack.inTensors.size());
  for (size_t i = 0; i < task.aclInTensors.size(); ++i) {
    task.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i));
  }

  DICP_LOG(INFO) << opName_ << " Create aclInTensor end";

  task.aclOutTensors.resize(variantPack.outTensors.size());
  for (size_t i = 0; i < task.aclOutTensors.size(); ++i) {
    task.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i));
  }

  DICP_LOG(INFO) << opName_ << " Create aclOutTensor end";
  DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
  return 0;
}

int AclNnAddRmsNormOperation::CallAclGetWorkspace(AclNnTask& task,
                                                  uint64_t& workspaceSize) {
  DICP_LOG(INFO) << opName_ << " aclnnAddRmsNormGetWorkspaceSize start";
  int ret = aclnnAddRmsNormGetWorkspaceSize(
      task.aclInTensors.at(0).tensor, task.aclInTensors.at(1).tensor,
      task.aclInTensors.at(2).tensor, this->epsilon,
      task.aclOutTensors.at(0).tensor, task.aclOutTensors.at(1).tensor,
      task.aclOutTensors.at(2).tensor, &workspaceSize, &task.aclExecutor);
  DICP_LOG(INFO) << opName_
                 << " aclnnAddRmsNormGetWorkspaceSize end, ret:" << ret
                 << ", workspaceSize:" << workspaceSize
                 << ", aclExecutor:" << task.aclExecutor;

  return ret;
}

int AclNnAddRmsNormOperation::CallAclExecute(uint8_t* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* aclExecutor,
                                             aclrtStream stream) {
  DICP_LOG(INFO) << opName_ << " aclnnAddRmsNorm start";
  int ret = aclnnAddRmsNorm(workspace, workspaceSize, aclExecutor, stream);
  DICP_LOG(INFO) << opName_ << " aclnnAddRmsNorm end, ret:" << ret;
  return ret;
}

AclNnTensor AclNnAddRmsNormOperation::CreateTensor(atb::Tensor atbTensor) {
  AclNnTensor aclNnTensor;
  aclNnTensor.needUpdateTensorDataPtr = true;
  aclNnTensor.atbTensor = atbTensor;
  return aclNnTensor;
}
}  // namespace dicp