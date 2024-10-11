#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <nlohmann/json.hpp>
#include <string>

#include "atb/operation.h"
#include "log.h"

namespace dicp {
constexpr size_t SVECTOR_SIZE = 8;

struct AclNnTensor {
  atb::Tensor atbTensor;
  atb::SVector<int64_t> strides;
  aclTensor* tensor = nullptr;
  bool needUpdateTensorDataPtr = false;
  int CreateTensor(const std::string& opName);
  int CreateTransposeTensor(const std::string& opName);
  int InitTensor(void* executor, const std::string& opName, const size_t index,
                 bool isInput);
};

struct AclNnTask {
  atb::SVector<AclNnTensor> aclInTensors;
  atb::SVector<aclIntArray*> aclInIntArrays;
  atb::SVector<AclNnTensor> aclOutTensors;
  aclOpExecutor* aclExecutor = nullptr;
  void Destory();
};

class AclNnOperation : public atb::Operation {
 public:
  explicit AclNnOperation(const std::string& name);
  ~AclNnOperation() override;
  std::string GetName() const override;
  atb::Status Setup(const atb::VariantPack& variantPack,
                    uint64_t& workspaceSize, atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace,
                      uint64_t workspaceSize, atb::Context* context) override;

 protected:
  aclTensor* CreateAclTensor(const AclNnTensor& aclNnTensor);
  atb::Status UpdateAclTensorDataPtr(const atb::VariantPack& variantPack);
  std::string opName_;
  AclNnTask aclNnTask_;

 private:
  virtual int CreateAclTensors(const atb::VariantPack& variantPack,
                               AclNnTask& task) = 0;
  virtual int CallAclGetWorkspace(AclNnTask& task, uint64_t& workspaceSize) = 0;
  virtual int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize,
                             aclOpExecutor* aclExecutor,
                             aclrtStream stream) = 0;
};
}  // namespace dicp
