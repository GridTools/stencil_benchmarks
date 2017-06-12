#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

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

    unsigned int isize=512;
    unsigned int jsize=512;
    unsigned int ksize=60;

    for (int i = 1; i < argc; i++)
    {
        if (!std::string("--isize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &isize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (!std::string("--jsize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &jsize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (!std::string("--ksize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &ksize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("Configuration :\n");
    printf("  isize: %d\n",isize);
    printf("  jsize: %d\n",jsize);
    printf("  ksize: %d\n",ksize);

    const size_t tot_size = isize*jsize*ksize;
    const size_t tsteps=100;
    const size_t warmup_step=10;
 
    printf("======================== FLOAT =======================\n");
    std::vector<double> timings(num_bench_st);
    std::fill(timings.begin(),timings.end(), 0);

    std::ofstream fs("./perf_results.json", std::ios_base::app);
    JSONNode globalNode;
    globalNode.cast(JSON_ARRAY);
    globalNode.set_name("metrics");

    JSONNode dsize;
    dsize.set_name("domain");

    dsize.push_back(JSONNode("x", isize));
    dsize.push_back(JSONNode("y", jsize));
    dsize.push_back(JSONNode("z", ksize));

    launch<float>(timings, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precf;
    precf.set_name("float");

    JSONNode fnotex;
    fnotex.set_name("no_tex");

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
  
      fnotex.push_back(stencils);
      precf.push_back(fnotex);

    }

    dsize.push_back(precf);

    printf("======================== DOUBLE =======================\n");

    std::fill(timings.begin(),timings.end(), 0);
    launch<double>(timings, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precd;
    precd.set_name("double");

    JSONNode dnotex;
    dnotex.set_name("no_tex");


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

      dnotex.push_back(stencils);
      precd.push_back(dnotex);

    }

    dsize.push_back(precd);

    globalNode.push_back(dsize);

    fs << globalNode.write_formatted() << std::endl;


}
