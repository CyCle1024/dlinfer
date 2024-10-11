#pragma once

#include <utility>

#include "log.h"
#include "model.h"
#include "nlohmann/json.hpp"

namespace dicp {

template <typename T>
T getValue(const nlohmann::json& node, const std::string& key) {
  try {
    return node.at(key).get<T>();
  } catch (const std::exception& e) {
    DICP_LOG(ERROR) << "Error: " << e.what();
    DICP_LOG(ERROR) << "JSON Node: " << node.dump(4);
    throw std::runtime_error("getValue failed!");
  }
}

class DICPModel : public Model {
 public:
  explicit DICPModel(const std::string& param);
  ~DICPModel();
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status InferShape(const std::vector<atb::TensorDesc>& inTensorDescs,
                         std::vector<atb::TensorDesc>& outTensorDescs) override;
  atb::Status PreProcess(const std::string& param) override;

 private:
  int64_t BuildGraph() override;
  void CreateSingleOperation(const nlohmann::json& paramJson, Node& node);
  void CreateGraphOperation(const nlohmann::json& paramJson, Node& node);
  atb::Status ParseParam(const std::string& param) override;
  atb::Status BindParamHostTensor(uint32_t nodeId) override;
  // atb::Status ParseParam(const std::string &param) override;
  std::vector<atb::TensorDesc> outputTensorDescs_;
  // atb::Status BindParamHostTensor(uint32_t nodeId) override;
  std::unordered_map<std::string, int> tensorsMap_;
  std::unordered_map<std::string, int> inputTensorsMap_;
  std::unordered_map<std::string, int> outputTensorsMap_;
  std::unordered_map<std::string, int> internalTensorsMap_;
  std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<int32_t>>>
      nodeHostTensorMap_;
};

}  // namespace dicp
