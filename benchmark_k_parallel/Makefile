include $(PWD)/Makefile.common

stencil_bench: main.cpp stencil_kernels.h tools.h defs.h
	$(compile) 

.PHONY: clean

clean:
	rm -f stencil_bench
