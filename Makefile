-include Makefile.user

PLATFORM_DEFS=-DPLATFORM_KNL

CCFLAGS=-std=c++11 -O1 -ggdb -qopenmp -ffreestanding -DNDEBUG -Isrc $(PLATFORM_DEFS) $(USERFLAGS)
LIBS=

%.o: %.cpp
	CC $(CCFLAGS) -c $< -o $@
	
stencil_bench: src/main.o src/arguments.o
	CC $(CCFLAGS) $+ $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f src/*.o stencil_bench

.PHONY: format
format:
	clang-format -i src/*.cpp src/*.h src/*/*.h
