include Makefile.user
include Makefile.config

STENCIL_KERNELS_FILE=stencil_kernels_$(STENCIL).h
STENCIL_KERNELS_FLAG=-DSTENCIL_KERNELS_H=\"$(STENCIL_KERNELS_FILE)\"
BLOCKSIZEX_FLAG=-DBLOCKSIZEX=$(BLOCKSIZEX)
BLOCKSIZEY_FLAG=-DBLOCKSIZEY=$(BLOCKSIZEY)
ALIGN_FLAG=-ALIGN=$(ALIGN)
LAYOUT_FLAG=-DLAYOUT=$(LAYOUT)

CONFIG_FLAGS=$(STENCIL_KERNELS_FLAG) $(BLOCKSIZEX_FLAG) $(BLOCKSIZEY_FLAG) $(ALIGN_FLAG) $(LAYOUT_FLAG)

stencil_bench: main.cpp tools.h defs.h $(STENCIL_KERNELS_FILE)
	CC $(CONFIG_FLAGS) -std=c++11 -O3 -qopenmp -DNDEBUG -DJSON_ISO_STRICT $(USERFLAGS) -Igridtools_storage/include -Ilibjson -Llibjson $< -ljson -lmemkind -o $@

.PHONY: clean

clean:
	rm -f stencil_bench
