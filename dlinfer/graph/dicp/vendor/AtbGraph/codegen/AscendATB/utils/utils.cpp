#include "utils.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "log.h"

namespace dicp {
namespace utils {

void* GetCurrentStream() {
  int32_t devId = 0;
  aclrtGetDevice(&devId);
  void* stream = c10_npu::getCurrentNPUStream(devId).stream();
  DICP_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
  return stream;
}

}  // namespace utils
}  // namespace dicp
