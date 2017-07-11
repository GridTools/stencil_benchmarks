.PHONY: benchmark_1D
.PHONY: benchmark_ij_parallel_i_first
.PHONY: benchmark_ij_parallel_k_first
.PHONY: benchmark_k_parallel
.PHONY: clean


all: benchmark_1D benchmark_ij_parallel_i_first benchmark_ij_parallel_k_first benchmark_k_parallel

benchmark_1D:
	$(MAKE) -C benchmark_1D

benchmark_ij_parallel_i_first:
	$(MAKE) -C benchmark_ij_parallel/i_first

benchmark_ij_parallel_k_first:
	$(MAKE) -C benchmark_ij_parallel/k_first

benchmark_k_parallel:
	$(MAKE) -C benchmark_k_parallel

clean:
	$(MAKE) -C benchmark_1D clean
	$(MAKE) -C benchmark_ij_parallel/i_first clean
	$(MAKE) -C benchmark_ij_parallel/k_first clean
	$(MAKE) -C benchmark_k_parallel clean

