#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#include "defs.h"
#include "libjson.h"
//#include "libjson/_internal/Source/JSONNode.h"

#include "stencil_kernels.h"

int parse_uint(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int main(int argc, char** argv) {

    unsigned int isize=256;
    unsigned int jsize=256;
    unsigned int ksize=80;

    bool write_out = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        std::cout << arg << std::endl;
        if (arg.find("--isize") != std::string::npos) {
            if (++i >= argc || !parse_uint(argv[i], &isize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (arg.find("--jsize") != std::string::npos) {
            if (++i >= argc || !parse_uint(argv[i], &jsize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (arg.find("--ksize") != std::string::npos) {
            if (++i >= argc || !parse_uint(argv[i], &ksize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (arg.find("--write") != std::string::npos) {
            write_out = true;
        }        
    }

    printf("Configuration :\n");
    printf("  isize: %d\n",isize);
    printf("  jsize: %d\n",jsize);
    printf("  ksize: %d\n",ksize);

    const size_t tot_size = isize*jsize*ksize-2*h;
    const size_t tsteps=100;
    const size_t warmup_step=10;
 
    printf("======================== FLOAT =======================\n");
    std::vector<double> timings(num_bench_st);
    std::fill(timings.begin(),timings.end(), 0);

    std::string s;
#ifdef CACHE_MODE
    s = "cache_mode";
#elif FLAT_MODE
    s = "flat_mode";
#endif

    std::stringstream ss, layout;
    layout << storage_info_t::layout_t::template at<0>() << "-" << storage_info_t::layout_t::template at<1>() << "-" << storage_info_t::layout_t::template at<2>();
    ss  << "./res_benchmark_1D_m" << s 
        << "_a" << ALIGN << "_l" 
        << layout.str() << "_t" 
        << omp_get_max_threads() << "_bsx" << BLOCKSIZEX << "_bsy" << BLOCKSIZEY << ".json";
        
    JSONNode globalNode;
    globalNode.cast(JSON_ARRAY);
    globalNode.set_name("metrics");

    JSONNode dsize;
    dsize.set_name("domain");

    dsize.push_back(JSONNode("x", isize));
    dsize.push_back(JSONNode("y", jsize));
    dsize.push_back(JSONNode("z", ksize));
    dsize.push_back(JSONNode("halo", h));
    dsize.push_back(JSONNode("alignment", ALIGN));
    dsize.push_back(JSONNode("layout", layout.str()));
    dsize.push_back(JSONNode("threads", omp_get_max_threads()));
    dsize.push_back(JSONNode("bsx", BLOCKSIZEX));
    dsize.push_back(JSONNode("bsy", BLOCKSIZEY));
#ifdef CACHE_MODE
    dsize.push_back(JSONNode("mode", "cache_mode"));
#elif FLAT_MODE
    dsize.push_back(JSONNode("mode", "flat_mode"));
#endif
    

    launch<float>(timings, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precf;
    precf.set_name("float");

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("avgi : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[avgi_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgi_st]);
    printf("avgj : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[avgj_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgj_st]);
    printf("avgk : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[avgk_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgk_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

    {
      JSONNode stencils;
      stencils.set_name("stencils");
      stencils.push_back(JSONNode("copy", tot_size*2*sizeof(float)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("copyi1", tot_size*2*sizeof(float)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumi1", tot_size*2*sizeof(float)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumj1", tot_size*2*sizeof(float)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumk1", tot_size*2*sizeof(float)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgi", tot_size*2*sizeof(float)/(timings[avgi_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgj", tot_size*2*sizeof(float)/(timings[avgj_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgk", tot_size*2*sizeof(float)/(timings[avgk_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("lap", tot_size*2*sizeof(float)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      precf.push_back(stencils);

    }

    dsize.push_back(precf);

    printf("======================== DOUBLE =======================\n");

    std::fill(timings.begin(),timings.end(), 0);
    launch<double>(timings, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precd;
    precd.set_name("double");

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("avgi : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[avgi_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgi_st]);
    printf("avgj : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[avgj_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgj_st]);
    printf("avgk : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[avgk_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[avgk_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

    {
      JSONNode stencils;
      stencils.set_name("stencils");
      stencils.push_back(JSONNode("copy", tot_size*2*sizeof(double)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("copyi1", tot_size*2*sizeof(double)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumi1", tot_size*2*sizeof(double)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumj1", tot_size*2*sizeof(double)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("sumk1", tot_size*2*sizeof(double)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgi", tot_size*2*sizeof(double)/(timings[avgi_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgj", tot_size*2*sizeof(double)/(timings[avgj_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("avgk", tot_size*2*sizeof(double)/(timings[avgk_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      stencils.push_back(JSONNode("lap", tot_size*2*sizeof(double)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.)));
      precd.push_back(stencils);

    }

    dsize.push_back(precd);

    globalNode.push_back(dsize);

    if(write_out) {
        std::ofstream fs(ss.str(), std::ios_base::app);
        fs << globalNode.write_formatted() << std::endl;
    }
}
