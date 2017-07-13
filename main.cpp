#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#include "tools.h"
#include "libjson.h"
#include STENCIL_KERNELS_H

int main(int argc, char** argv) {

    unsigned int isize=256;
    unsigned int jsize=256;
    unsigned int ksize=80;

    // Read flags
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

    const storage_info_t si(isize-2*h, jsize-2*h, ksize-2*h);
    int begin = si.index(h,h,h);
    int end = si.index(isize+h,jsize+h,ksize+h);
    const size_t tot_size = end-begin;
    const size_t tsteps=22;
    const size_t warmup_step=1;

    printf("Configuration :\n");
    printf("  isize: %d\n",isize);
    printf("  jsize: %d\n",jsize);
    printf("  ksize: %d\n",ksize);
    printf("  touched elements: %d\n", tot_size);
      
    printf("======================== FLOAT =======================\n");
    timing times;

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

    auto get_float_bw = [&](std::string const& s) -> double {
        auto size = tot_size*2*sizeof(float);
        auto mean = times.mean(s);
        return size/mean/(1024.*1024.*1024.);
    };

    auto get_double_bw = [&](std::string const& s) -> double {
        auto size = tot_size*2*sizeof(double);
        auto mean = times.mean(s);
        return size/mean/(1024.*1024.*1024.);
    };

    auto push_stencil_info = [&](JSONNode& par, std::string const& name, double bw) {
            JSONNode stenc;
            stenc.set_name(name);
            stenc.push_back(JSONNode("bw", bw));
            stenc.push_back(JSONNode("runs", times.size(name)));
            stenc.push_back(JSONNode("rms", times.rms(name)));
            stenc.push_back(JSONNode("sum", times.sum(name)));
            stenc.push_back(JSONNode("median", times.median(name)));
            stenc.push_back(JSONNode("mean", times.mean(name)));
            stenc.push_back(JSONNode("min", times.min(name)));            
            stenc.push_back(JSONNode("max", times.max(name)));            
            par.push_back(stenc);
    };

    launch<float>(times, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precf;
    precf.set_name("float");

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", get_float_bw("copy"),      times.sum("copy"));
    printf("copyi1 : %f GB/s, time : %f \n", get_float_bw("copyi1"),  times.sum("copyi1"));
    printf("sumi1 : %f GB/s, time : %f \n", get_float_bw("sumi1"),    times.sum("sumi1"));
    printf("sumj1 : %f GB/s, time : %f \n", get_float_bw("sumj1"),    times.sum("sumj1"));
    printf("sumk1 : %f GB/s, time : %f \n", get_float_bw("sumk1"),    times.sum("sumk1"));
    printf("avgi : %f GB/s, time : %f \n", get_float_bw("avgi"),      times.sum("avgi"));
    printf("avgj : %f GB/s, time : %f \n", get_float_bw("avgj"),      times.sum("avgj"));
    printf("avgk : %f GB/s, time : %f \n", get_float_bw("avgk"),      times.sum("avgk"));
    printf("lap : %f GB/s, time : %f \n", get_float_bw("lap"),        times.sum("lap"));

    {
      JSONNode stencils;
      stencils.set_name("stencils");
      push_stencil_info(stencils, "copy", get_float_bw("copy"));
      push_stencil_info(stencils, "copyi1", get_float_bw("copyi1"));
      push_stencil_info(stencils, "sumi1", get_float_bw("sumi1"));
      push_stencil_info(stencils, "sumj1", get_float_bw("sumj1"));
      push_stencil_info(stencils, "sumk1", get_float_bw("sumk1"));
      push_stencil_info(stencils, "avgi", get_float_bw("avgi"));
      push_stencil_info(stencils, "avgj", get_float_bw("avgj"));
      push_stencil_info(stencils, "avgk", get_float_bw("avgk"));
      push_stencil_info(stencils, "lap", get_float_bw("lap"));
      precf.push_back(stencils);
    }

    dsize.push_back(precf);

    printf("======================== DOUBLE =======================\n");

    times.clear();
    launch<double>(times, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precd;
    precd.set_name("double");

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", get_double_bw("copy"),      times.sum("copy"));
    printf("copyi1 : %f GB/s, time : %f \n", get_double_bw("copyi1"),  times.sum("copyi1"));
    printf("sumi1 : %f GB/s, time : %f \n", get_double_bw("sumi1"),    times.sum("sumi1"));
    printf("sumj1 : %f GB/s, time : %f \n", get_double_bw("sumj1"),    times.sum("sumj1"));
    printf("sumk1 : %f GB/s, time : %f \n", get_double_bw("sumk1"),    times.sum("sumk1"));
    printf("avgi : %f GB/s, time : %f \n", get_double_bw("avgi"),      times.sum("avgi"));
    printf("avgj : %f GB/s, time : %f \n", get_double_bw("avgj"),      times.sum("avgj"));
    printf("avgk : %f GB/s, time : %f \n", get_double_bw("avgk"),      times.sum("avgk"));
    printf("lap : %f GB/s, time : %f \n", get_double_bw("lap"),        times.sum("lap"));

    {
      JSONNode stencils;
      stencils.set_name("stencils");
      push_stencil_info(stencils, "copy", get_double_bw("copy"));
      push_stencil_info(stencils, "copyi1", get_double_bw("copyi1"));
      push_stencil_info(stencils, "sumi1", get_double_bw("sumi1"));
      push_stencil_info(stencils, "sumj1", get_double_bw("sumj1"));
      push_stencil_info(stencils, "sumk1", get_double_bw("sumk1"));
      push_stencil_info(stencils, "avgi", get_double_bw("avgi"));
      push_stencil_info(stencils, "avgj", get_double_bw("avgj"));
      push_stencil_info(stencils, "avgk", get_double_bw("avgk"));
      push_stencil_info(stencils, "lap", get_double_bw("lap"));
      precd.push_back(stencils);
    }

    dsize.push_back(precd);

    globalNode.push_back(dsize);

    if(write_out) {
        std::ofstream fs(ss.str(), std::ios_base::app);
        fs << globalNode.write_formatted() << std::endl;
    }
}
