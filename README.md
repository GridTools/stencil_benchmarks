# Stencil Benchmarks

A collection of basic stencil (and related) benchmarks, tuned for various
computing architectures. Based on Python, generates and runs optimized
C/C++/CUDA/HIP codes.

## Installation

Using pip:

```bash
$> git clone https://github.com/MeteoSwiss-APN/stencil_benchmarks.git
$> pip install stencil_benchmarks
```

## Basic Usage

Running the original STREAM benchmark (on a CPU) is as simple as this, giving
the results (here of a bad old laptop) in a formatted table:

```bash
$> sbench stream mc-calpin original
       bandwidth  avg-time      time  max-time  ticks                                                         
name                                                 
copy     10526.9  0.020382  0.015199  0.031769  14018
scale    10527.0  0.020908  0.015199  0.029451  14018
add      11935.0  0.027315  0.020109  0.038648  14018
triad    11890.6  0.028889  0.020184  0.056173  14018
```

For GPUs, a STREAM implementation also exists, just use `sbench stream cuda-hip
native --compiler nvcc`. Always use the `--help` flag, it provides an overview
over each commands flags and sub-commands (so check `sbench --help`, `sbench
stream --help`, etc.).
