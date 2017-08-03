-include Makefile.user

CCFLAGS=-std=c++11 -O3 -g -DNDEBUG -Isrc $(USERFLAGS)

SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:.cpp=.o)

SRCS_KNL=$(wildcard src/knl/*.cpp)
OBJS_KNL=$(SRCS_KNL:.cpp=.o)

%.o: %.cpp
	CC $(CCFLAGS) -c $< -o $@
	
stencil_bench_knl: CCFLAGS+=-DPLATFORM_KNL -ffreestanding -qopenmp
stencil_bench_knl: $(OBJS) $(OBJS_KNL)
	CC $(CCFLAGS) $+ -o $@

.PHONY: clean
clean:
	rm -f src/*.o src/*/*.o stencil_bench_*

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.cpp src/*/*.h