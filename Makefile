-include Makefile.user

CCFLAGS=-std=c++11 -O3 -g -MMD -MP -Wall -fopenmp -DNDEBUG -Isrc $(USERFLAGS)
NVCCFLAGS=-std=c++11 -arch=sm_60 -O3 -g -DNDEBUG -Isrc $(USERFLAGS_CUDA)

SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:.cpp=.o)
DEPS=$(SRCS:.cpp=.d)

SRCS_KNL=$(wildcard src/knl/*.cpp)
OBJS_KNL=$(SRCS_KNL:.cpp=.o)
DEPS_KNL=$(SRCS_KNL:.cpp=.d)

SRCS_CUDA=$(wildcard src/cuda/*.cu)
OBJS_CUDA=$(SRCS_CUDA:.cu=.o)

%.o: %.cpp
	CC $(CCFLAGS) -c $< -o $@

%.o: %.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

.PHONY: knl
knl: stencil_bench_knl

.PHONY: cuda
cuda: stencil_bench_cuda
	
stencil_bench_knl: CCFLAGS+=-DPLATFORM_KNL -ffreestanding
stencil_bench_knl: $(OBJS) $(OBJS_KNL)
	CC $(CCFLAGS) $+ -o $@

stencil_bench_cuda: CCFLAGS+=-DPLATFORM_CUDA
stencil_bench_cuda: CUFLAGS+=-DPLATFORM_CUDA
stencil_bench_cuda: $(OBJS) $(OBJS_CUDA)
	nvcc $(NVCCFLAGS) $+ -o $@

-include $(DEPS) $(DEPS_KNL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(OBJS_KNL) $(DEPS_KNL) $(OBJS_CUDA) $(DEPS_CUDA) stencil_bench_knl stencil_bench_cuda

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.cpp src/*/*.h src/*/*.cu
