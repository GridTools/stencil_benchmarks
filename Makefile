-include Makefile.user

CCFLAGS=-std=c++11 -O3 -g -MMD -MP -Wall -DNDEBUG -Isrc $(USERFLAGS)

SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:.cpp=.o)
DEPS=$(SRCS:.cpp=.d)

SRCS_KNL=$(wildcard src/knl/*.cpp)
OBJS_KNL=$(SRCS_KNL:.cpp=.o)
DEPS_KNL=$(SRCS_KNL:.cpp=.d)

%.o: %.cpp
	CC $(CCFLAGS) -c $< -o $@
	
stencil_bench_knl: CCFLAGS+=-DPLATFORM_KNL -ffreestanding -qopenmp
stencil_bench_knl: $(OBJS) $(OBJS_KNL)
	CC $(CCFLAGS) $+ -o $@

-include $(DEPS) $(DEPS_KNL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(OBJS_KNL) $(DEPS_KNL) stencil_bench_knl

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.cpp src/*/*.h
