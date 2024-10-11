#pragma once

#include <memory>
#include <atb/context.h>

namespace dicp {
class ContextFactory {
public:
    static std::shared_ptr<atb::Context> GetAtbContext(void *stream);
    static void FreeAtbContext();
};

} // namespace dicp
