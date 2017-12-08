-include Makefile.user

CXXFLAGS=-std=c++11 -O3 -MMD -MP -Wall -fopenmp -DNDEBUG -Isrc $(USERFLAGS)
NVCXXFLAGS=-std=c++11 -arch=sm_60 -O3 -g -DNDEBUG -Isrc $(USERFLAGS_CUDA)
LIBS=$(USERLIBS)

CXXFLAGS_KNL=-DPLATFORM_KNL
LIBS_KNL=
CXXFLAGS_CUDA=-DPLATFORM_CUDA
LIBS_CUDA=

CXXVERSION=$(shell $(CXX) --version)
ifneq (,$(findstring icpc,$(CXXVERSION)))
	CXXFLAGS_KNL+=-xmic-avx512 -ffreestanding
else ifneq (,$(findstring g++,$(CXXVERSION)))
	LIBS+=-lgomp
	CXXFLAGS+=-Wno-unknown-pragmas -Wno-unused-variable
	CXXFLAGS_KNL+=-march=knl -mtune=knl -fvect-cost-model=unlimited
	LIBS_KNL+=-lmemkind
endif


SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:src/%.cpp=%.o)
DEPS=$(SRCS:.cpp=.d)

SRCS_KNL=$(wildcard src/knl/*.cpp)
OBJS_KNL=$(SRCS_KNL:src/knl/%.cpp=%.o)
DEPS_KNL=$(SRCS_KNL:.cpp=.d)

SRCS_CUDA=$(wildcard src/cuda/*.cu)
OBJS_CUDA=$(SRCS_CUDA:.cu=.o)	

%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: src/knl/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_KNL) -c $< -o $@

%.o: src/cuda/%.cu
	nvcc $(NVCXXFLAGS) $(CXXFLAGS_CUDA) -c $< -o $@

.PHONY: default
default:
	$(error Please specify the target platform, i.e. use 'make knl' for Intel KNL or 'make cuda' for NVIDIA CUDA GPUs)

.PHONY: knl
knl: $(OBJS) $(OBJS_KNL)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_KNL) $+ $(LIBS) $(LIBS_KNL) -o stencil_bench

.PHONY: cuda
cuda: $(OBJS) $(OBJS_CUDA)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_CUDA) $+ $(LIBS) $(LIBS_CUDA) -o stencil_bench

-include $(DEPS) $(DEPS_KNL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(OBJS_KNL) $(DEPS_KNL) $(OBJS_CUDA) $(DEPS_CUDA) $(OBJS_X86) $(DEPS_X86) stencil_bench

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.cpp src/*/*.h src/*/*.cu
