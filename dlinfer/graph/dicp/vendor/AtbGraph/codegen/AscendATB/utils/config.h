#pragma once

#include <atb/types.h>

namespace dicp {

class Config {
 public:
  Config();
  ~Config(){};
  bool ReuseInternalTensor();
  bool UseTilingCopyStream();
  bool IsTaskQueueEnable();
  bool IsConvertNCHWToND();
  bool UsePlanExecuteAsync();
  uint64_t WorkspaceBufferSize();
  void PrintConfig() const;

 private:
  bool reuseInternalTensor_;
  bool useTilingCopyStream_;
  bool isTaskQueueEnable_;
  bool isConvertNCHWToND_;
  bool isUsePlanExecuteAsync_;
  uint64_t workspaceBufferSize_;
};

Config& GetConfig();

}  // namespace dicp