-include Makefile.user

OBJDIR=build
CXXFLAGS=-std=c++11 -MMD -O3 -MP -Wall -fopenmp -DNDEBUG -Isrc $(USERFLAGS)
NVCCFLAGS=-std=c++11 -arch=sm_60 -O3 -g -Xcompiler -fopenmp -DNDEBUG -Isrc $(USERFLAGS_CUDA)
LIBS=$(USERLIBS)

CXXFLAGS_TX2=-DPLATFORM_TX2
LIBS_TX2=
CXXFLAGS_KNL=-DPLATFORM_KNL
LIBS_KNL=
CXXFLAGS_CUDA=-DPLATFORM_CUDA
LIBS_CUDA=
CXXFLAGS_KNLCPU=-DPLATFORM_KNL -DKNL_NO_HBWMALLOC -march=native -mtune=native -mcpu=native
LIBS_KNLCPU=

CXXVERSION=$(shell $(CXX) --version)
ifneq (,$(findstring icpc,$(CXXVERSION)))
	CXXFLAGS_KNL+=-xmic-avx512 -ffreestanding
	CXXFLAGS_KNLCPU+=-ffreestanding
else ifneq (,$(findstring g++,$(CXXVERSION)))
	LIBS+=-lgomp 
	CXXFLAGS+=-Wno-unknown-pragmas -Wno-unused-variable 
	CXXFLAGS_KNL+=-march=knl -mtune=knl -fvect-cost-model=unlimited
	CXXFLAGS_KNLCPU+=-fvect-cost-model=unlimited
	CXXFLAGS_TX2+=-fvect-cost-model=unlimited -DKNL_NO_HBWMALLOC -march=native -mtune=native -mcpu=native
	LIBS_KNL+=-lmemkind
endif


SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:src/%.cpp=$(OBJDIR)/%.o)
DEPS=$(SRCS:src/%.cpp=$(OBJDIR)/%.d)

SRCS_KNL=$(wildcard src/knl/*.cpp)
OBJS_KNL=$(SRCS_KNL:src/knl/%.cpp=$(OBJDIR)/%.o)
DEPS_KNL=$(SRCS_KNL:src/knl/%.cpp=$(OBJDIR)/%.d)

SRCS_CUDA=$(wildcard src/cuda/*.cu)
OBJS_CUDA=$(SRCS_CUDA:src/cuda/%.cu=$(OBJDIR)/%.o)

$(OBJDIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: src/knl/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: src/cuda/%.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: default
default:
	$(error Please specify the target platform, i.e. use 'make knl' for Intel KNL, \
		'make cuda' for NVIDIA CUDA GPUs, 'make tx2' for ThunderX2  or 'make knl-cpu' \
		to compile the KNL implementation for common CPUs)

.PHONY: knl
knl: sbench_knl

.PHONY: cuda
cuda: sbench_cuda

.PHONY: knl-cpu
knl-cpu: sbench_knlcpu

.PHONY: knl-tx2
knl-tx2: sbench_tx2

sbench_knl: CXXFLAGS+=$(CXXFLAGS_KNL)
sbench_knl: $(OBJS) $(OBJS_KNL)
	$(CXX) $(CXXFLAGS) $+ $(LIBS) $(LIBS_KNL) -o $@

sbench_cuda: CXX=nvcc
sbench_cuda: CXXFLAGS=$(NVCCFLAGS) $(CXXFLAGS_CUDA)
sbench_cuda: $(OBJS) $(OBJS_CUDA)
	$(CXX) $(CXXFLAGS) $+ $(LIBS) $(LIBC_CUDA) -o $@

sbench_knlcpu: CXXFLAGS+=$(CXXFLAGS_KNLCPU)
sbench_knlcpu: $(OBJS) $(OBJS_KNL)
	$(CXX) $(CXXFLAGS) $+ $(LIBS) $(LIBS_KNLCPU) -o $@

sbench_tx2: CXXFLAGS+=$(CXXFLAGS_TX2)
sbench_tx2: $(OBJS) $(OBJS_KNL)
	$(CXX) $(CXXFLAGS) $+ $(LIBS) $(LIBS_TX2) -o $@

-include $(DEPS) $(DEPS_KNL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(OBJS_KNL) $(DEPS_KNL) $(OBJS_CUDA) sbench_*

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.cpp src/*/*.h src/*/*.cu
