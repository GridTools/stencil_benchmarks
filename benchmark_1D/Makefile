stencil_bench: main.cpp stencil_kernels.h tools.h defs.h
	CC main.cpp --std=c++11 -DNDEBUG -I/users/stefanm/boost_1_62_0/include -I../gridtools_storage/include -I../libjson -DJSON_ISO_STRICT -L../libjson -ljson  -lmemkind  -O3 -fopenmp -o stencil_bench

clean: 
	rm -f stencil_bench
