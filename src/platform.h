#pragma once

#include <memory>

#include "arguments.h"
#include "variant_base.h"

namespace platform {

void setup(arguments& args);

std::unique_ptr<variant_base> create_variant(const arguments_map& args);

}  // namespace platform
