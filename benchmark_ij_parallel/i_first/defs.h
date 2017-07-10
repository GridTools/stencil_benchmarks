#pragma once

#ifndef BLOCKSIZEX
    #define BLOCKSIZEX 8
#endif

#ifndef BLOCKSIZEY
    #define BLOCKSIZEY 8
#endif

#ifndef ALIGN
    #define ALIGN 16
#endif

#ifndef LAYOUT
    #define LAYOUT 2,1,0
#endif

enum stencils { copy_st=0, copyi1_st, sumi1_st, sumj1_st, sumk1_st, avgi_st, avgj_st, avgk_st, lap_st, num_bench_st };

