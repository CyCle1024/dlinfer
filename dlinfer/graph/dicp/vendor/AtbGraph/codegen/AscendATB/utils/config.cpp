#include "config.h"

#include <cstdlib>
#include <string>

#include "log.h"

namespace dicp {

constexpr int GB_1 = 1024 * 1024 * 1024;

Config::Config() {
  const char* envTensorReuse = std::getenv("DICP_INTERNAL_TENSOR_REUSE");
  if (envTensorReuse == nullptr || std::string(envTensorReuse) == "1") {
    reuseInternalTensor_ = true;
  } else {
    reuseInternalTensor_ = false;
  }

  const char* envUseTillingCopyStream =
      std::getenv("DICP_USE_TILLING_COPY_STREAM");
  if (envUseTillingCopyStream == nullptr ||
      std::string(envUseTillingCopyStream) == "0") {
    useTilingCopyStream_ = false;
  } else {
    useTilingCopyStream_ = true;
  }

  const char* taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
  const char* blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
  bool isTaskQueueEnable_ =
      !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
        (blockingEnv != nullptr && std::string(blockingEnv) == "1"));

  const char* envBufferSize = std::getenv("DICP_WORKSPACE_BUFFER_SIZE");
  if (envBufferSize) {
    workspaceBufferSize_ = std::stoull(envBufferSize);
  } else {
    workspaceBufferSize_ = 1 * GB_1;
  }

  const char* envConvertNCHWToND = std::getenv("DICP_CONVERT_NCHW_TO_ND");
  if (envTensorReuse == nullptr || std::string(envTensorReuse) == "1") {
    isConvertNCHWToND_ = true;
  } else {
    isConvertNCHWToND_ = false;
  }

  const char* envUsePlanExecuteAsync =
      std::getenv("ATB_OPERATION_EXECUTE_ASYNC");
  isUsePlanExecuteAsync_ = (envUsePlanExecuteAsync != nullptr &&
                            std::string(envUsePlanExecuteAsync) == "1");
}

bool Config::ReuseInternalTensor() { return reuseInternalTensor_; }

bool Config::IsTaskQueueEnable() { return isTaskQueueEnable_; }

bool Config::IsConvertNCHWToND() { return isConvertNCHWToND_; }

bool Config::UseTilingCopyStream() { return useTilingCopyStream_; }

bool Config::UsePlanExecuteAsync() { return isUsePlanExecuteAsync_;}

uint64_t Config::WorkspaceBufferSize() { return workspaceBufferSize_; }

void Config::PrintConfig() const {
  DICP_LOG(INFO) << "layerIsLayerInternalTensorReuse_: "
                 << (reuseInternalTensor_ ? "True" : "False");
}

Config& GetConfig() {
  static Config config;
  return config;
}

}  // namespace dicp
