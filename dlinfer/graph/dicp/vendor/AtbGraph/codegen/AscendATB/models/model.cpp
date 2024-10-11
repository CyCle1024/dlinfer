#include "model.h"

#include <acl/acl.h>
#include <atb/types.h>
#include <atb/utils.h>

#include <nlohmann/json.hpp>

#include "config.h"
#include "log.h"
#include "tensor_utils.h"

namespace dicp {
static bool IsTensorDimsEqual(const atb::Dims& left, const atb::Dims& other) {
  if (left.dimNum != other.dimNum) {
    return false;
  }

  for (uint64_t i = 0; i < left.dimNum; ++i) {
    if (left.dims[i] != other.dims[i]) {
      return false;
    }
  }

  return true;
}

std::string Model::Graph::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < inTensors.size(); ++i) {
    ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " "
       << tensor_utils::TensorToString(inTensors.at(i)) << std::endl;
  }
  for (size_t i = 0; i < outTensors.size(); ++i) {
    ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " "
       << tensor_utils::TensorToString(outTensors.at(i)) << std::endl;
  }
  for (size_t i = 0; i < internalTensors.size(); ++i) {
    ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " "
       << tensor_utils::TensorToString(internalTensors.at(i)) << std::endl;
  }
  ss << "nodes:" << nodes.size() << std::endl;

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto& node = nodes.at(i);
    ss << "node[" << i << "] operation:" << node.operation.get()
       << ", operationName:" << node.operation->GetName() << std::endl;
    for (auto tensorIt : node.inTensors) {
      ss << "node[" << i << "] inTensor:" << tensorIt << " "
         << tensor_utils::TensorToString(*tensorIt) << std::endl;
    }
    for (auto tensorIt : node.outTensors) {
      ss << "node[" << i << "] outTensor:" << tensorIt << " "
         << tensor_utils::TensorToString(*tensorIt) << std::endl;
    }
  }
  return ss.str();
}

void Model::Graph::Init() {
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes.at(i);
    node.variantPack.inTensors.resize(node.inTensors.size());
    node.variantPack.outTensors.resize(node.outTensors.size());
  }
  InitTensorType();
  InitTensorMaxNodeMap();
}

void Model::Graph::InitTensorType() {
  for (auto& node : nodes) {
    node.inTensorTypes.resize(node.inTensors.size());
    node.outTensorTypes.resize(node.outTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
      node.inTensorTypes.at(i) = IsInternalTensor(node.inTensors.at(i))
                                     ? Model::INTERMEDIATE_TENSOR
                                     : Model::NOT_INTERMEDIATE_TENSOR;
    }
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
      node.outTensorTypes.at(i) = IsInternalTensor(node.outTensors.at(i))
                                      ? Model::INTERMEDIATE_TENSOR
                                      : Model::NOT_INTERMEDIATE_TENSOR;
    }
  }
}

bool Model::Graph::IsInternalTensor(const atb::Tensor* tensor) {
  for (auto& internalTensor : internalTensors) {
    if (&internalTensor == tensor) {
      return true;
    }
  }

  return false;
}

void Model::Graph::InitTensorMaxNodeMap() {
  std::map<atb::Tensor*, uint64_t> tensorMaxNodeIdMap;
  maxNodeIdTensorMap.clear();

  for (size_t i = 0; i < internalTensors.size(); ++i) {
    atb::Tensor& internalTensor = internalTensors[i];
    uint64_t maxNodeId = 0;
    uint64_t dependNodeCount = 0;
    for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
      auto& node = nodes.at(nodeId);
      for (auto inTensorIt : node.inTensors) {
        if (&internalTensor == inTensorIt) {
          maxNodeId = nodeId;
          dependNodeCount++;
        }
      }
    }
    tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
    DICP_LOG_IF(dependNodeCount == 0, ERROR)
        << "runner graph internal tensor[" << i
        << "] dependNodeCount is 0, graph wrong";
    maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
  }
}

Model::Model(const std::string& modelName, const std::string& param)
    : modelName_(modelName), param_(param) {
  aclrtGetDevice(&currentDevId_);
}

Model::~Model() {}

int64_t Model::Init(
    GetWorkspaceFunc getWorkSpaceFunc,
    CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
    RunTaskFunc runTaskFunc) {
  isUsePlanExecuteAsync_ = GetConfig().UsePlanExecuteAsync();
  if (isUsePlanExecuteAsync_ && !runTaskFunc) {
    std::thread thread =
        std::thread(std::bind(&Model::ThreadProcessTask, this));
    taskProcessThread_ = std::move(thread);
  }

  DICP_LOG(INFO) << modelName_
                  << " new, isTaskQueueEnable:" << (runTaskFunc != nullptr)
                  << ", isUsePlanExecuteAsync:" << isUsePlanExecuteAsync_
                  << ", currentDevId:" << currentDevId_;

  getWorkSpaceFunc_ = getWorkSpaceFunc;
  createTensorFromTensorDescFunc_ = createTensorFromTensorDescFunc;
  runTaskFunc_ = runTaskFunc;

  int64_t atbStatus = BuildGraph();
  graph_.Init();
  DICP_LOG(DEBUG) << modelName_ << " init graph:\n" << graph_.ToString();
  return atbStatus;
}

atb::Status Model::Execute(atb::Context* context,
                           std::vector<atb::Tensor>& inTensors,
                           std::vector<atb::Tensor>& outTensors,
                           const std::string& param) {
  if (graph_.inTensors.size() != inTensors.size() ||
      graph_.outTensors.size() != outTensors.size()) {
    DICP_LOG(ERROR) << modelName_
                    << " graph.inTensors.size:" << graph_.inTensors.size()
                    << ", inTensors.size:" << inTensors.size()
                    << ", graph.outTensors.size:" << graph_.outTensors.size()
                    << ", outTensors.size:" << outTensors.size();
    return atb::ERROR_INVALID_GRAPH;
  }

  ParseParam(param);

  ClearInternalTensors();
  nodeOutTensors_.clear();

  allTaskFinish_ = false;
  context_ = context;
  graph_.inTensors = inTensors;
  graph_.outTensors = outTensors;
  DICP_LOG(INFO) << modelName_
                 << " execute start, executeCount:" << executeCount_
                 << ", graph:\n"
                 << graph_.ToString();

  for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
    BuildNodeVariantPack(nodeId);
    BindParamHostTensor(nodeId);
    atb::Status st = ExecuteNode(nodeId);
    if (st != 0) {
      return st;
    }
  }

  WaitAsyncPlanExecuteFinish();

  DICP_LOG(INFO) << modelName_ << " executeCount:" << executeCount_;

  executeCount_++;

  return atb::NO_ERROR;
}

atb::Status Model::ParseParam(const std::string& param) {
  (void)param;
  return atb::NO_ERROR;
}

atb::Status Model::PreProcess(const std::string& param) {
  (void)param;
  return atb::NO_ERROR;
}

atb::Status Model::BindParamHostTensor(uint32_t nodeId) {
  (void)nodeId;
  return atb::NO_ERROR;
}

void Model::BuildNodeVariantPack(int nodeId) {
  auto& node = graph_.nodes.at(nodeId);
  bool needReshape = node.inTensorReshapeFuncs.size() > 0;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.variantPack.inTensors.size());
  inTensorDescs.resize(node.variantPack.inTensors.size());
  for (size_t i = 0; i < node.inTensors.size(); ++i) {
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
    inTensorDescs.at(i) = node.inTensors.at(i)->desc;
    if (needReshape) {
      node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape,
                                      inTensorDescs.at(i).shape);
      node.variantPack.inTensors.at(i).desc.shape = inTensorDescs.at(i).shape;
      node.inTensors.at(i)->desc.shape = inTensorDescs.at(i).shape;
    }
    DICP_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] inTensors[" << i
                   << "]:"
                   << tensor_utils::TensorToString(
                          node.variantPack.inTensors.at(i));
  }

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
  DICP_LOG_IF(st != 0, FATAL) << modelName_ << " nodes[" << nodeId << "] "
                              << " infer shape fail, error code: " << st;

  for (size_t i = 0; i < node.outTensors.size(); ++i) {
    node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
    if (node.outTensorTypes.at(i) == Model::INTERMEDIATE_TENSOR) {
      node.variantPack.outTensors.at(i) = MallocInternalTensor(
          node.outTensors.at(i), nodeId, i, outTensorDescs.at(i));
      *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
    }
  }

  auto it = graph_.maxNodeIdTensorMap.find(nodeId);
  if (it != graph_.maxNodeIdTensorMap.end()) {
    for (auto tensorIt : it->second) {
      FreeInternalTensor(tensorIt->deviceData);
    }
  }
}

atb::Status Model::ExecuteNode(int nodeId) {
  auto& node = graph_.nodes.at(nodeId);

  Timer timer;
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    DICP_LOG(ERROR) << modelName_ << " setup node[" << nodeId
                    << "] fail, not call execute";
    return st;
  }

  DICP_LOG(INFO) << modelName_ << " get node[" << nodeId
                 << "] workspace size:" << node.workspaceSize;

  if (node.workspaceSize > 0) {
    node.workspace = getWorkSpaceFunc_(node.workspaceSize);
  }

  if (isUsePlanExecuteAsync_) {
    Timer timer;
    ExecutePlanAsync(nodeId);
  } else {
    st = ExecutePlanSync(nodeId);
  }
  return st;
}

void Model::ThreadProcessTask() {
  DICP_LOG(INFO) << modelName_ << " thread process operations start";
  int ret = aclrtSetDevice(currentDevId_);
  DICP_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

  size_t processTaskCount = 0;
  while (true) {
    int nodeId = PopTask();
    atb::Status st = ExecutePlanSync(nodeId);
    if (st != 0) {
      allTaskFinish_ = true;
      processTaskCount = 0;
      return;
    }
    processTaskCount++;
    if (processTaskCount == graph_.nodes.size()) {
      DICP_LOG(INFO) << modelName_ << " thread process all operations";
      processTaskCount = 0;
      allTaskFinish_ = true;
    }
  }
}

atb::Status Model::ExecutePlanSync(int nodeId) {
  auto& node = graph_.nodes.at(nodeId);
  atb::VariantPack& variantPack = node.variantPack;

  DICP_LOG(INFO) << modelName_ << "execute node[" << nodeId << "] start";

  atb::Status st = node.operation->Execute(
      variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
  if (st != 0) {
    DICP_LOG(ERROR) << "execute node[" << nodeId
                    << "] fail, error code: " << st;
  }
  return st;
}

void Model::ExecutePlanAsync(int nodeId) {
  if (runTaskFunc_) {
    runTaskFunc_(modelName_ + std::to_string(nodeId), [=]() {
      ExecutePlanSync(nodeId);
      return 0;
    });
  } else {
    PushTask(nodeId);
  }
}

void Model::PushTask(int nodeId) {
  std::unique_lock<std::mutex> lock(mutex_);
  taskQueue_.push(nodeId);
  lock.unlock();
  cond_.notify_one();
}

int Model::PopTask() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (taskQueue_.empty()) {
    cond_.wait(lock);
  }
  int nodeId = taskQueue_.front();
  taskQueue_.pop();
  return nodeId;
}

void Model::WaitAsyncPlanExecuteFinish() {
  if (isUsePlanExecuteAsync_ && !runTaskFunc_) {
    while (true) {
      if (allTaskFinish_) {
        DICP_LOG(INFO) << modelName_ << " allTaskFinish is true, break";
        break;
      }
    }
  }
}

void Model::ExecuteNodeView(int nodeId) {
  auto& node = graph_.nodes.at(nodeId);
  if (node.inTensorReshapeFuncs.size() > 0) {
    for (size_t i = 0; i < node.inTensors.size() &&
                       node.inTensorReshapeFuncs.at(i) != nullptr;
         i++) {
      node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape,
                                      node.inTensors.at(i)->desc.shape);
    }
  }
}

bool Model::IsTensorDescEqual(const atb::TensorDesc& tensorDesc,
                              const atb::Tensor& atbTensor) const {
  return atbTensor.desc.dtype == tensorDesc.dtype &&
         atbTensor.desc.format == tensorDesc.format &&
         IsTensorDimsEqual(atbTensor.desc.shape, tensorDesc.shape);
}

void Model::ClearInternalTensors() { internalTensors_.clear(); }

atb::Tensor Model::MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId,
                                        size_t outTensorId,
                                        const atb::TensorDesc& tensorDesc) {
  if (GetConfig().ReuseInternalTensor()) {
    std::vector<atb::Tensor*>::iterator iter =
        std::find(nodeOutTensors_.begin(), nodeOutTensors_.end(), outTensor);
    if (iter != nodeOutTensors_.end()) {
      DICP_LOG(INFO) << modelName_ << " nodeId: " << nodeId
                     << ", out tensor id: " << outTensorId << " write inplace";
      return **iter;
    }
    for (auto& it : internalTensors_) {
      if (it.second) {  // Tensor被使用中，不能被分配其他Op
        continue;
      }

      if (IsTensorDescEqual(tensorDesc, it.first)) {
        it.second = true;
        DICP_LOG(INFO) << modelName_ << " use old internal tensor";
        return it.first;
      }
    }
  }

  DICP_LOG(INFO) << modelName_ << " create internal tensor, node[" << nodeId
                 << "], outTensor[" << outTensorId << "]";
  atb::Tensor newTensor = createTensorFromTensorDescFunc_(tensorDesc);
  internalTensors_.push_back(std::make_pair(newTensor, true));
  nodeOutTensors_.push_back(outTensor);
  return newTensor;
}

void Model::FreeInternalTensor(void* tensorDeviceData) {
  if (GetConfig().ReuseInternalTensor()) {
    for (auto& it : internalTensors_) {
      if (it.first.deviceData == tensorDeviceData) {
        it.second = false;  // Tensor被释放，可以被后来者使用
        DICP_LOG(INFO) << modelName_ << " free internal tensor";
        break;
      }
    }
  }
}

void Model::GetModelTensorNameList(
    nlohmann::json& modelJson,
    std::map<atb::Tensor*, std::string>& tensorNameMap) {
  std::string tensorName;

  for (size_t i = 0; i < graph_.inTensors.size(); i++) {
    tensorName = modelName_ + "_input_" + std::to_string(i);
    modelJson["inTensors"].emplace_back(tensorName);
    atb::Tensor& inTensor = graph_.inTensors[i];
    tensorNameMap[&inTensor] = tensorName;
  }

  for (size_t i = 0; i < graph_.outTensors.size(); i++) {
    tensorName = modelName_ + "_output_" + std::to_string(i);
    modelJson["outTensors"].emplace_back(tensorName);
    atb::Tensor& outTensor = graph_.outTensors[i];
    tensorNameMap[&outTensor] = tensorName;
  }

  for (size_t i = 0; i < graph_.internalTensors.size(); i++) {
    tensorName = modelName_ + "_internal_" + std::to_string(i);
    modelJson["internalTensors"].emplace_back(tensorName);
    atb::Tensor& internalTensor = graph_.internalTensors[i];
    tensorNameMap[&internalTensor] = tensorName;
  }
}

void Model::GetNodeTopoInfo(
    nlohmann::json& nodeJson, const Node& opNode,
    const std::map<atb::Tensor*, std::string> tensorNameMap) {
  nodeJson["opName"] = opNode.operation->GetName();

  for (auto inTensor : opNode.inTensors) {
    auto it = tensorNameMap.find(inTensor);
    if (it != tensorNameMap.end()) {
      nodeJson["inTensors"].emplace_back(it->second);
    }
  }

  for (auto outTensor : opNode.outTensors) {
    auto it = tensorNameMap.find(outTensor);
    if (it != tensorNameMap.end()) {
      nodeJson["outTensors"].emplace_back(it->second);
    }
  }
}

std::string Model::GetModelTopoInfo() {
  nlohmann::json modelJson;
  modelJson["modelName"] = modelName_;

  std::map<atb::Tensor*, std::string> tensorNameMap;
  GetModelTensorNameList(modelJson, tensorNameMap);

  for (size_t nodeId = 0; nodeId < graph_.nodes.size(); nodeId++) {
    const auto& opNode = graph_.nodes.at(nodeId);
    nlohmann::json nodeJson;
    GetNodeTopoInfo(nodeJson, opNode, tensorNameMap);
    modelJson["nodes"].emplace_back(nodeJson);
  }
  return modelJson.dump();
}

}  // namespace dicp
