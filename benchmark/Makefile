stencil_bench: main.cu stencil_kernels.h tools.h
	icc main.cpp --std=c++11 -I/users/stefanm/boost_1_62_0/include -I../gridtools_storage/include -I./libjson -DJSON_ISO_STRICT -L./libjson -ljson -O3 -fopenmp -o stencil_kernels

clean: 
	rm -f stencil_bench
