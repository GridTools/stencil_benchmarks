#include <iomanip>
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
    std::string out_name;
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
          if (++i >= argc) {
            std::cerr << "Wrong parsing" << std::endl;
          }
          write_out = true;
          out_name = argv[i];
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
      
    std::cout << "======================== FLOAT =======================" << std::endl;
    timing times;

    std::stringstream layout;
    layout << storage_info_t::layout_t::template at<0>() << "-"
           << storage_info_t::layout_t::template at<1>() << "-"
           << storage_info_t::layout_t::template at<2>();
        
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
        auto time = times.min(s);
        return size/time/(1024.*1024.*1024.);
    };

    auto get_double_bw = [&](std::string const& s) -> double {
        auto size = tot_size*2*sizeof(double);
        auto time = times.min(s);
        return size/time/(1024.*1024.*1024.);
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

          std::cout << std::setw(6) << name << ": "
                    << std::setw(7) << bw << " GB/s (max), time: "
                    << std::setw(12) << times.mean(name) << "s (avg), "
                    << std::setw(12) << times.min(name) << "s (min), "
                    << std::setw(12) << times.max(name) << "s (max)" << std::endl;
    };

    launch<float>(times, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precf;
    precf.set_name("float");

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

    std::cout << "======================== DOUBLE =======================" << std::endl;

    times.clear();
    launch<double>(times, isize, jsize, ksize, tsteps, warmup_step);

    JSONNode precd;
    precd.set_name("double");

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
        std::ofstream fs(out_name, std::ios_base::app);
        fs << globalNode.write_formatted() << std::endl;
    }
}
