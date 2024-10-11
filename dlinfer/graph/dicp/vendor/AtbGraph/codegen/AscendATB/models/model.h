#pragma once
#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "log.h"
#include "timer.h"

namespace dicp {
class Model {
 public:
  using ReshapeFunc =
      std::function<void(const atb::Dims& oldDims, atb::Dims& newDims)>;
  using GetWorkspaceFunc = std::function<void*(uint64_t bufferSize)>;
  using CreateTensorFromTensorDescFunc =
      std::function<atb::Tensor(const atb::TensorDesc& tensorDesc)>;
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;
  enum TensorType {
    INTERMEDIATE_TENSOR = 0,
    NOT_INTERMEDIATE_TENSOR,
  };

  struct Node {
    std::shared_ptr<atb::Operation> operation;
    std::vector<atb::Tensor*> inTensors;
    std::vector<atb::Tensor*> outTensors;
    atb::VariantPack variantPack;
    // std::vector<torch::Tensor> torchTensors;
    atb::SVector<ReshapeFunc> inTensorReshapeFuncs;
    atb::SVector<TensorType> inTensorTypes;
    atb::SVector<TensorType> outTensorTypes;
    uint64_t workspaceSize = 0;
    void* workspace = nullptr;
  };

  struct Graph {
    std::vector<atb::Tensor> inTensors;
    std::vector<atb::Tensor> outTensors;
    std::vector<atb::Tensor> internalTensors;
    std::vector<Node> nodes;
    std::map<uint64_t, std::set<atb::Tensor*>> maxNodeIdTensorMap;
    void Init();
    std::string ToString() const;

   private:
    void InitTensorType();
    bool IsInternalTensor(const atb::Tensor* tensor);
    void InitTensorMaxNodeMap();
  };

  Model(const std::string& modelName, const std::string& param);
  virtual ~Model();
  int64_t Init(GetWorkspaceFunc getWorkSpaceFunc,
               CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
               RunTaskFunc runTaskFunc = nullptr);

  virtual uint32_t GetInputNum() const = 0;
  virtual uint32_t GetOutputNum() const = 0;
  virtual atb::Status InferShape(
      const std::vector<atb::TensorDesc>& inTensorDescs,
      std::vector<atb::TensorDesc>& outTensorDescs) = 0;
  atb::Status Execute(atb::Context* context,
                      std::vector<atb::Tensor>& inTensors,
                      std::vector<atb::Tensor>& outTensors,
                      const std::string& param);

 public:
  virtual atb::Status PreProcess(const std::string& param);

 protected:
  virtual int64_t BuildGraph() = 0;
  virtual atb::Status ParseParam(const std::string& param);
  virtual atb::Status BindParamHostTensor(uint32_t nodeId);

 protected:
  bool IsTensorDescEqual(const atb::TensorDesc& tensorDesc,
                         const atb::Tensor& atTensor) const;
  void ExecuteNodeView(int nodeId);
  void BuildNodeVariantPack(int nodeId);
  atb::Status ExecuteNode(int nodeId);
  void ThreadProcessTask();
  atb::Status ExecutePlanSync(int nodeId);
  void ExecutePlanAsync(int nodeId);
  void PushTask(int nodeId);
  int PopTask();
  void WaitAsyncPlanExecuteFinish();
  void ClearInternalTensors();
  atb::Tensor MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId,
                                   size_t outTensorId,
                                   const atb::TensorDesc& tensorDesc);
  void FreeInternalTensor(void* tensorDeviceData);
  void GetModelTensorNameList(
      nlohmann::json& modelJson,
      std::map<atb::Tensor*, std::string>& tensorNameMap);
  void GetNodeTopoInfo(nlohmann::json& nodeJson, const Node& opNode,
                       const std::map<atb::Tensor*, std::string> tensorNameMap);
  std::string GetModelTopoInfo();

 protected:
  GetWorkspaceFunc getWorkSpaceFunc_;
  CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc_;
  RunTaskFunc runTaskFunc_ = nullptr;
  std::string modelName_;
  std::string param_;
  Graph graph_;

  uint64_t executeCount_ = 0;
  atb::Context* context_;

  bool isUsePlanExecuteAsync_ = false;
  std::queue<int> taskQueue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread taskProcessThread_;
  std::atomic_bool allTaskFinish_;
  int32_t currentDevId_ = 0;
  std::vector<std::pair<atb::Tensor, bool>> internalTensors_;
  std::vector<atb::Tensor*> nodeOutTensors_;
};
}  // namespace dicp
