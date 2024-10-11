
#include "context_factory.h"

#include <thread>

#include "config.h"
#include "log.h"
#include "utils.h"

namespace dicp {
thread_local std::shared_ptr<atb::Context> g_localContext;

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext(void* stream) {
  if (g_localContext) {
    DICP_LOG(INFO) << "ContextFactory return localContext";
    return g_localContext;
  }
  DICP_LOG(INFO) << "ContextFactory create atb::Context start";
  atb::Context* context = nullptr;
  atb::Status st = atb::CreateContext(&context);
  DICP_LOG_IF(st != 0, ERROR) << "ContextFactory create atb::Context fail";

  if (context) {
    context->SetExecuteStream(stream);
    if (GetConfig().UseTilingCopyStream()) {
      DICP_LOG(INFO) << "ContextFactory use tiling copy stream";
      context->SetAsyncTilingCopyStatus(true);
    } else {
      DICP_LOG(INFO) << "ContextFactory not use tiling copy stream";
    }
  }

  std::shared_ptr<atb::Context> tmpLocalContext(
      context, [](atb::Context* context) { atb::DestroyContext(context); });
  g_localContext = tmpLocalContext;

  return g_localContext;
}

void ContextFactory::FreeAtbContext() {
  DICP_LOG(INFO) << "ContextFactory FreeAtbContext start.";
  if (!g_localContext) {
    return;
  }

  DICP_LOG(INFO) << "ContextFactory localContext use_count: "
                 << g_localContext.use_count();
  if (g_localContext.use_count() != 1) {
    return;
  }
  DICP_LOG(INFO) << "ContextFactory localContext reset.";
  g_localContext.reset();
}
}  // namespace dicp