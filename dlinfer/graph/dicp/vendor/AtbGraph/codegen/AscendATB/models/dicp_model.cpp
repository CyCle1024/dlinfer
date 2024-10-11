#include "dicp_model.h"

#include <fstream>

#include "operation_creator.h"
#include "tensor_utils.h"

namespace dicp {

DICPModel::DICPModel(const std::string& param) : Model("DICPModel", param) {}

DICPModel::~DICPModel() {}

uint32_t DICPModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DICPModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DICPModel::PreProcess(const std::string& param) {
  auto paramJson = nlohmann::json::parse(param);
  int cur = 0;
  for (const auto& out : paramJson["outputTensorDescs"]) {
    if (out.contains("format")) {
      auto format = getValue<int32_t>(out, "format");
      outputTensorDescs_[cur].format = static_cast<aclFormat>(format);
    } else {
      outputTensorDescs_[cur].format = ACL_FORMAT_ND;
    }
    auto dtype = getValue<int32_t>(out, "dtype");
    outputTensorDescs_[cur].dtype = static_cast<aclDataType>(dtype);
    auto dimNum = getValue<int32_t>(out, "dimNum");
    outputTensorDescs_[cur].shape.dimNum = static_cast<uint64_t>(dimNum);
    auto tmp_dims = getValue<std::vector<int64_t>>(out, "dims");
    for (int i = 0; i < dimNum; ++i) {
      outputTensorDescs_[cur].shape.dims[i] = tmp_dims[i];
    }
    cur++;
  }
  return atb::NO_ERROR;
}

atb::Status DICPModel::InferShape(
    [[maybe_unused]] const std::vector<atb::TensorDesc>& inTensorDescs,
    [[maybe_unused]] std::vector<atb::TensorDesc>& outTensorDescs) {
  DICP_LOG(INFO) << "Enter dicp DICPModel InferShape";
  if (outTensorDescs.size() != GetOutputNum()) {
    return atb::ERROR_INVALID_GRAPH;
  }
  for (unsigned int i = 0; i < outTensorDescs.size(); ++i) {
    outTensorDescs[i].format = outputTensorDescs_[i].format;
    outTensorDescs[i].dtype = outputTensorDescs_[i].dtype;
    outTensorDescs[i].shape.dimNum = outputTensorDescs_[i].shape.dimNum;
    for (size_t j = 0; j < outTensorDescs[i].shape.dimNum; ++j) {
      outTensorDescs[i].shape.dims[j] = outputTensorDescs_[i].shape.dims[j];
    }
    DICP_LOG(INFO) << "######## DICPModel outTensorDescs[" << i << "]:"
                   << tensor_utils::TensorDescToString(outTensorDescs.at(i));
  }
  return atb::NO_ERROR;
}

void DICPModel::CreateSingleOperation(const nlohmann::json& paramJson,
                                      Node& node) {
  auto opType = getValue<std::string>(paramJson, "type");
  auto opName = getValue<std::string>(paramJson, "name");
  auto opInputNames =
      getValue<std::vector<std::string>>(paramJson, "inputNames");
  auto opOutputNames =
      getValue<std::vector<std::string>>(paramJson, "outputNames");
  atb::Operation* op = CreateOperation(opType, paramJson["param"]);

  node.operation.reset(op);
  for (const auto& t : opInputNames) {
    if (inputTensorsMap_.count(t) > 0) {
      node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
    } else if (internalTensorsMap_.count(t) > 0) {
      node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
    } else if (outputTensorsMap_.count(t) > 0) {
      node.inTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
    } else {
      DICP_LOG(ERROR) << "cannot find name in input/internal: " << t;
      throw std::runtime_error("cannot find name in input/internal!");
    }
  }
  for (const auto& t : opOutputNames) {
    if (outputTensorsMap_.count(t) > 0) {
      node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
    } else if (internalTensorsMap_.count(t) > 0) {
      node.outTensors.push_back(
          &graph_.internalTensors[internalTensorsMap_[t]]);
    } else {
      DICP_LOG(ERROR) << "cannot find name in output/internal: " << t;
      throw std::runtime_error("cannot find name in input/internal!");
    }
  }
  bool hasReshapeInputs = getValue<bool>(paramJson, "hasReshapeInputs");
  if (hasReshapeInputs) {
    node.inTensorReshapeFuncs.resize(node.inTensors.size());
    auto reshapeInputs = paramJson["reshapeInputs"];
    int count = 0;
    for (const auto& reshapeInput : paramJson["reshapeInputs"]) {
      auto reshapeType = getValue<std::string>(reshapeInput, "reshapeType");
      if (reshapeType == "None") {
        node.inTensorReshapeFuncs.at(count) = [=](const atb::Dims& oldShape,
                                                  atb::Dims& newShape) {
          newShape = oldShape;
        };
      } else if (reshapeType == "view") {
        auto dimNum = getValue<int32_t>(reshapeInput, "dimNum");
        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dims");
        bool needInferDim = false;
        size_t dimNeedInfer = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
          if (dims[i] == -1) {
            needInferDim = true;
            dimNeedInfer = i;
            break;
          }
        }
        node.inTensorReshapeFuncs.at(count) =
            [=]([[maybe_unused]] const atb::Dims& oldShape,
                atb::Dims& newShape) {
              newShape.dimNum = dimNum;
              if (needInferDim) {
                int64_t totalValue = 1;
                int64_t otherProd = 1;
                for (size_t i = 0; i < oldShape.dimNum; ++i) {
                  totalValue *= oldShape.dims[i];
                }
                for (size_t i = 0; i < dims.size(); ++i) {
                  if (i != dimNeedInfer) {
                    otherProd *= dims[i];
                  }
                }
                newShape.dims[dimNeedInfer] = totalValue / otherProd;
              }
              for (size_t i = 0; i < dims.size(); ++i) {
                if (dims[i] == -1) continue;
                newShape.dims[i] = dims[i];
              }
            };
      } else if (reshapeType == "unsqueeze") {
        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
        node.inTensorReshapeFuncs.at(count) =
            [=]([[maybe_unused]] const atb::Dims& oldShape,
                atb::Dims& newShape) {
              std::vector<int64_t> dimValues;
              dimValues.resize(oldShape.dimNum);
              for (size_t i = 0; i < oldShape.dimNum; ++i) {
                dimValues[i] = oldShape.dims[i];
              }
              for (size_t i = 0; i < dims.size(); ++i) {
                auto pos = dimValues.begin() + dims[i];
                dimValues.insert(pos, 1);
              }
              for (size_t i = 0; i < dimValues.size(); ++i) {
                newShape.dims[i] = dimValues[i];
              }
              newShape.dimNum = dimValues.size();
            };
      } else if (reshapeType == "squeeze") {
        auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
        node.inTensorReshapeFuncs.at(count) =
            [=]([[maybe_unused]] const atb::Dims& oldShape,
                atb::Dims& newShape) {
              std::vector<int64_t> dimValues;
              dimValues.resize(oldShape.dimNum);
              for (size_t i = 0; i < oldShape.dimNum; ++i) {
                dimValues[i] = oldShape.dims[i];
              }
              for (size_t i = 0; i < dims.size(); ++i) {
                auto pos = dimValues.begin() + dims[i];
                dimValues.erase(pos);
              }
              for (size_t i = 0; i < dimValues.size(); ++i) {
                newShape.dims[i] = dimValues[i];
              }
              newShape.dimNum = dimValues.size();
            };
      }
      count++;
    }
  }
}

void DICPModel::CreateGraphOperation(const nlohmann::json& paramJson,
                                     Node& node) {
  atb::GraphParam graph_param;
  int nodeSize = getValue<int32_t>(paramJson, "nodeSize");
  auto inputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
  auto outputNames =
      getValue<std::vector<std::string>>(paramJson, "outputNames");
  auto internalNames =
      getValue<std::vector<std::string>>(paramJson, "internalNames");
  graph_param.inTensorNum = inputNames.size();
  graph_param.outTensorNum = outputNames.size();
  graph_param.internalTensorNum = internalNames.size();
  graph_param.nodes.resize(nodeSize);

  // graph local tensor ids
  std::unordered_map<std::string, int> graph_tensor_ids;
  int tensorCount = 0;
  for (unsigned int i = 0; i < inputNames.size(); ++i) {
    graph_tensor_ids[inputNames[i]] = tensorCount++;
  }
  for (unsigned int i = 0; i < outputNames.size(); ++i) {
    graph_tensor_ids[outputNames[i]] = tensorCount++;
  }
  for (unsigned int i = 0; i < internalNames.size(); ++i) {
    graph_tensor_ids[internalNames[i]] = tensorCount++;
  }

  int cur_node_index = 0;
  for (const auto& node : paramJson["nodes"]) {
    auto nodeType = getValue<std::string>(node, "nodeType");
    auto nodeOp = node["value"];
    if (nodeType == "singleOperation") {
      auto opType = getValue<std::string>(nodeOp, "type");
      auto opName = getValue<std::string>(nodeOp, "name");
      auto opInputNames =
          getValue<std::vector<std::string>>(nodeOp, "inputNames");
      auto opOutputNames =
          getValue<std::vector<std::string>>(nodeOp, "outputNames");
      atb::Operation* op = CreateOperation(opType, nodeOp["param"]);
      graph_param.nodes[cur_node_index].operation = op;
      for (const auto& t : opInputNames) {
        graph_param.nodes[cur_node_index].inTensorIds.push_back(
            graph_tensor_ids[t]);
      }
      for (const auto& t : opOutputNames) {
        graph_param.nodes[cur_node_index].outTensorIds.push_back(
            graph_tensor_ids[t]);
      }

      bool hasReshapeInputs = getValue<bool>(nodeOp, "hasReshapeInputs");
      auto& cur_node = graph_param.nodes[cur_node_index];
      if (hasReshapeInputs) {
        cur_node.inTensorReshapeFuncs.resize(cur_node.inTensorIds.size());
        auto reshapeInputs = nodeOp["reshapeInputs"];
        int count = 0;
        for (const auto& reshapeInput : nodeOp["reshapeInputs"]) {
          auto reshapeType = getValue<std::string>(reshapeInput, "reshapeType");
          if (reshapeType == "None") {
            cur_node.inTensorReshapeFuncs.at(count) =
                [=](const atb::Dims& oldShape, atb::Dims& newShape) {
                  newShape = oldShape;
                };
          } else if (reshapeType == "view") {
            auto dimNum = getValue<int32_t>(reshapeInput, "dimNum");
            auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dims");
            bool needInferDim = false;
            size_t dimNeedInfer = 0;
            for (size_t i = 0; i < dims.size(); ++i) {
              if (dims[i] == -1) {
                needInferDim = true;
                dimNeedInfer = i;
                break;
              }
            }
            cur_node.inTensorReshapeFuncs.at(count) =
                [=]([[maybe_unused]] const atb::Dims& oldShape,
                    atb::Dims& newShape) {
                  newShape.dimNum = dimNum;
                  if (needInferDim) {
                    int64_t totalValue = 1;
                    int64_t otherProd = 1;
                    for (size_t i = 0; i < oldShape.dimNum; ++i) {
                      totalValue *= oldShape.dims[i];
                    }
                    for (size_t i = 0; i < dims.size(); ++i) {
                      if (i != dimNeedInfer) {
                        otherProd *= dims[i];
                      }
                    }
                    newShape.dims[dimNeedInfer] = totalValue / otherProd;
                  }
                  for (size_t i = 0; i < dims.size(); ++i) {
                    if (dims[i] == -1) continue;
                    newShape.dims[i] = dims[i];
                  }
                };
          } else if (reshapeType == "unsqueeze") {
            auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
            cur_node.inTensorReshapeFuncs.at(count) =
                [=]([[maybe_unused]] const atb::Dims& oldShape,
                    atb::Dims& newShape) {
                  std::vector<int64_t> dimValues;
                  dimValues.resize(oldShape.dimNum);
                  for (size_t i = 0; i < oldShape.dimNum; ++i) {
                    dimValues[i] = oldShape.dims[i];
                  }
                  for (size_t i = 0; i < dims.size(); ++i) {
                    auto pos = dimValues.begin() + dims[i];
                    dimValues.insert(pos, 1);
                  }
                  for (size_t i = 0; i < dimValues.size(); ++i) {
                    newShape.dims[i] = dimValues[i];
                  }
                  newShape.dimNum = dimValues.size();
                };
          } else if (reshapeType == "squeeze") {
            auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
            cur_node.inTensorReshapeFuncs.at(count) =
                [=]([[maybe_unused]] const atb::Dims& oldShape,
                    atb::Dims& newShape) {
                  std::vector<int64_t> dimValues;
                  dimValues.resize(oldShape.dimNum);
                  for (size_t i = 0; i < oldShape.dimNum; ++i) {
                    dimValues[i] = oldShape.dims[i];
                  }
                  for (size_t i = 0; i < dims.size(); ++i) {
                    auto pos = dimValues.begin() + dims[i];
                    dimValues.erase(pos);
                  }
                  for (size_t i = 0; i < dimValues.size(); ++i) {
                    newShape.dims[i] = dimValues[i];
                  }
                  newShape.dimNum = dimValues.size();
                };
          }
          count++;
        }
      }
    } else {
      DICP_LOG(ERROR) << "invalid node type in graph opearation, ndoeType: "
                      << nodeType;
      throw std::runtime_error("invalid node type in graph opearation!");
    }
    cur_node_index++;
  }

  auto hasInferShape = getValue<bool>(paramJson, "hasInferShape");
  if (hasInferShape) {
    auto& inferShape = paramJson["inferShape"];
    auto inferType = getValue<std::string>(inferShape, "type");
    if (inferType == "equal") {
      auto outputByInput = getValue<std::vector<int32_t>>(inferShape, "value");
      graph_param.inferShapeFunc =
          [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
              atb::SVector<atb::TensorDesc>& outTensorDescs) {
            for (size_t i = 0; i < outTensorDescs.size(); ++i) {
              outTensorDescs.at(i) = inTensorDescs.at(outputByInput[i]);
            }
            return atb::NO_ERROR;
          };
    }
  }

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(graph_param, &op);
  if (st != 0) {
    DICP_LOG(ERROR) << "atb CreateOperation graph failed, st: " << st;
    throw std::runtime_error("atb CreateOperation graph failed!");
  }

  // bind Model tensor ids to graph tensor
  node.operation.reset(op);
  for (const auto& t : inputNames) {
    if (inputTensorsMap_.count(t) > 0) {
      node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
    } else {
      node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
    }
  }
  for (const auto& t : outputNames) {
    if (outputTensorsMap_.count(t) > 0) {
      node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
    } else {
      node.outTensors.push_back(
          &graph_.internalTensors[internalTensorsMap_[t]]);
    }
  }
}

int64_t DICPModel::BuildGraph() {
  // get json
  std::ifstream f(param_);
  auto paramJson = nlohmann::json::parse(f);

  // parse json
  auto graphInputNames =
      getValue<std::vector<std::string>>(paramJson, "inputNames");
  auto graphOutputNames =
      getValue<std::vector<std::string>>(paramJson, "outputNames");
  auto graphInternalNames =
      getValue<std::vector<std::string>>(paramJson, "internalNames");

  int tensorCount = 0;
  graph_.inTensors.resize(graphInputNames.size());
  graph_.outTensors.resize(graphOutputNames.size());
  graph_.internalTensors.resize(graphInternalNames.size());
  outputTensorDescs_.resize(graphOutputNames.size());
  for (unsigned int i = 0; i < graphInputNames.size(); ++i) {
    if (tensorsMap_.count(graphInputNames[i]) > 0) {
      DICP_LOG(ERROR) << "duplicate tensor name: " << graphInputNames[i];
      throw std::runtime_error("duplicate tensor name!");
    }
    tensorsMap_[graphInputNames[i]] = tensorCount++;
    inputTensorsMap_[graphInputNames[i]] = i;
  }
  for (unsigned int i = 0; i < graphOutputNames.size(); ++i) {
    if (tensorsMap_.count(graphOutputNames[i]) > 0) {
      DICP_LOG(ERROR) << "duplicate tensor name: " << graphOutputNames[i];
      throw std::runtime_error("duplicate tensor name");
    }
    tensorsMap_[graphOutputNames[i]] = tensorCount++;
    outputTensorsMap_[graphOutputNames[i]] = i;
  }
  for (unsigned int i = 0; i < graphInternalNames.size(); ++i) {
    if (tensorsMap_.count(graphInternalNames[i]) > 0) {
      DICP_LOG(ERROR) << "duplicate tensor name: " << graphInternalNames[i];
      throw std::runtime_error("duplicate tensor name");
    }
    tensorsMap_[graphInternalNames[i]] = tensorCount++;
    internalTensorsMap_[graphInternalNames[i]] = i;
  }

  for (const auto& node : paramJson["nodes"]) {
    auto nodeType = getValue<std::string>(node, "nodeType");
    auto nodeOp = node["value"];
    Node cur_node;

    if (nodeType == "singleOperation") {
      CreateSingleOperation(nodeOp, cur_node);
    } else if (nodeType == "graphOperation") {
      CreateGraphOperation(nodeOp, cur_node);
    } else {
      DICP_LOG(ERROR) << "invalid node type: " << nodeType;
      throw std::runtime_error("invalid node type!");
    }

    graph_.nodes.push_back(cur_node);
  }

  for (const auto& hostTensor : paramJson["hostTensorNames"]) {
    auto nodeId = getValue<int32_t>(hostTensor, "nodeId");
    auto tensorId = getValue<int32_t>(hostTensor, "tensorId");
    nodeHostTensorMap_[nodeId][tensorId] = {};
  }

  DICP_LOG(INFO) << "DICPModel BuildGraph success";
  return atb::NO_ERROR;
}

atb::Status DICPModel::ParseParam(const std::string& param) {
  nlohmann::json paramJson = nlohmann::json::parse(param);
  for (const auto& node : paramJson["hostTensors"]) {
    auto nodeId = getValue<int32_t>(node, "nodeId");
    auto tensorId = getValue<int32_t>(node, "tensorId");
    auto value = getValue<std::vector<int32_t>>(node, "value");
    nodeHostTensorMap_[nodeId][tensorId] = value;
  }
  return atb::NO_ERROR;
}

atb::Status DICPModel::BindParamHostTensor(uint32_t nodeId) {
  if (nodeHostTensorMap_.count(nodeId) == 0) {
    return atb::NO_ERROR;
  }

  auto& node = graph_.nodes.at(nodeId);
  for (auto& i : nodeHostTensorMap_[nodeId]) {
    node.variantPack.inTensors.at(i.first).hostData = i.second.data();
  }
  DICP_LOG(INFO) << "BindParamHostTensor end";
  return atb::NO_ERROR;
}

}  // namespace dicp
