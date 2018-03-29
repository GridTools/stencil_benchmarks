#pragma once

#ifdef PLATFORM_KNL
#include "knl/knl_platform.h"
#elif defined(PLATFORM_TX2)
#include "knl/knl_platform.h"
#elif defined(PLATFORM_CUDA)
#include "cuda/cuda_platform.h"
#else
#error "No platform defined."
#endif
