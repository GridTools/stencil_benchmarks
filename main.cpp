#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#include <omp.h>
#include <boost/align/is_aligned.hpp>
#include <storage/storage-facility.hpp>

#ifdef FLAT_MODE
#include <hbwmalloc.h>
#endif

#include "tools.h"
#include "libjson.h"

constexpr int h = 2;
typedef gridtools::storage_traits<gridtools::enumtype::Host> storage_tr;
typedef storage_tr::custom_layout_storage_info_t<0, gridtools::layout_map< LAYOUT >, gridtools::halo< h,h,h >, gridtools::alignment< ALIGN > > storage_info_t;

#include STENCIL_KERNELS_H


template<typename T>
void launch( timing& times, const unsigned int is, const unsigned int js, const unsigned ks, const unsigned tsteps, const unsigned warmup_step ) {
    int isize = is-2*h;
    int jsize = js-2*h;
    int ksize = ks-2*h;

    const storage_info_t si(isize, jsize, ksize);

    std::cout << "Size: " << isize << " " << jsize << " " << ksize << std::endl;
    std::cout << "Storage size (including halo+padding): "
              << si.template dim<0>() << " " << si.template dim<1>() << " "
              << si.template dim<2>() << std::endl;
    std::cout << "Halo: " << h << std::endl;
    std::cout << "Alignment: " << storage_info_t::alignment_t::value << std::endl;
    std::cout << "Initial offset: " << si.get_initial_offset() << std::endl;
    std::cout << "Total size: " << si.size() << std::endl;
    std::cout << "Storage Size: " << si.template dim<0>() << " " << si.template dim<1>()
              << " " << si.template dim<2>() << std::endl;
    std::cout << "Zero pos: " << si.index(0,0,0) << std::endl;
    std::cout << "Zero pos+halo: " << si.index(h,h,h) << std::endl;

#ifdef CACHE_MODE
    T* a = (T*)aligned_alloc(ALIGN*sizeof(T), si.size()*sizeof(T));
    T* b = (T*)aligned_alloc(ALIGN*sizeof(T), si.size()*sizeof(T));
#elif FLAT_MODE
    T* a = (T*) hbw_malloc(si.size()*sizeof(T));
    T* b = (T*) hbw_malloc(si.size()*sizeof(T));
#else
    static_assert(false, "Please define either FLAT_MODE or CACHE_MODE.");
#endif

    auto check_alignment = [](const std::string& s, const T* ptr) {
      bool aligned = boost::alignment::is_aligned(ptr, ALIGN * sizeof(T));
      std::cout << "\t" << s << ": " << ptr << " ---> " << (aligned ? "yes" : "no") << std::endl;
      return aligned;
    };
#define CHECK_ALIGNMENT(x) check_alignment(#x, x)

    //check if aligned to ALIGN*sizeof(T) bytes
    std::cout << "Alignment checks:" << std::endl;
    if (!CHECK_ALIGNMENT(a + si.index(h, h, h)) ||
        !CHECK_ALIGNMENT(b + si.index(h, h, h))) {
      std::cerr << "First data point not aligned..." << std::endl;
      std::abort();
    }

    CHECK_ALIGNMENT(a);
    CHECK_ALIGNMENT(b);

    for(unsigned int i=0; i < isize+2*h; ++i) {
        for(unsigned int j=0; j < jsize+2*h; ++j) {
            for(unsigned int k=0; k < ksize+2*h; ++k) {
                a[si.index(i,j,k)] = si.index(i,j,k);
            }
        }
    }

    std::chrono::high_resolution_clock::time_point t1,t2;
    cache_flusher c;

    for(unsigned int t=0; t < tsteps; t++) {

        //----------------------------------------//
        //----------------  COPY  ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        copy(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("copy", std::chrono::duration<double>(t2-t1).count());
        if(!t) {
            for(unsigned int i=h; i < isize+h; ++i) {
                for(unsigned int j=h; j < jsize+h; ++j) {
                    for(unsigned int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //---------------- COPYi1 ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        copyi1(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("copyi1", std::chrono::duration<double>(t2-t1).count());
        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i+1,j,k)] ) {
                            printf("Error1 in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMi1 ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumi1(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumi1", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i+1,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMj1 ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumj1(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumj1", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i,j+1,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j+1,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMk1 ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumk1(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumk1", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i,j,k+1)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)], a[si.index(i,j,k+1)]);
                        }
                    }
                }
            }
        }


        //----------------------------------------//
        //----------------  AVGi  ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgi(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgi", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i-1,j,k)] + a[si.index(i+1,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i-1,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  AVGj  ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgj(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgj", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)],
                                    a[si.index(i,j-1,k)], a[si.index(i,j+1,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  AVGk  ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgk(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgk", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k-1)] + a[si.index(i,j,k+1)]) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)],
                                    a[si.index(i,j,k-1)], a[si.index(i,j,k+1)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  LAP   ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        lap(a, b, si, isize, jsize, ksize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("lap", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize+h; ++i) {
                for(int j=h; j < jsize+h; ++j) {
                    for(int k=h; k < ksize+h; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i-1,j,k)] +
                               a[si.index(i+1,j,k)] + a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)] ) {
                            auto res =a[si.index(i,j,k)] + a[si.index(i-1,j,k)] +
                               a[si.index(i+1,j,k)] + a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)];
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], res);
                        }
                    }
                }
            }
        }
    }

#ifdef CACHE_MODE
    free(a);
    free(b);
#elif FLAT_MODE
    hbw_free(a);
    hbw_free(b);
#endif

}

int main(int argc, char** argv) {

    unsigned int isize=256;
    unsigned int jsize=256;
    unsigned int ksize=80;

    // Read flags
    bool write_out = false;
    std::string out_name;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg(argv[i]);
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
    const size_t tot_size = (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h);
    const size_t tsteps=22;
    const size_t warmup_step=1;

    printf("Configuration :\n");
    printf("  isize: %d\n",isize);
    printf("  jsize: %d\n",jsize);
    printf("  ksize: %d\n",ksize);
    printf("  touched elements: %d\n", tot_size);
    printf("  linear elements: %d\n", si.index(isize + h, jsize + h, ksize + h) - si.index(h, h, h));
      
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
#ifdef BLOCKSIZEX
    dsize.push_back(JSONNode("bsx", BLOCKSIZEX));
#endif
#ifdef BLOCKSIZEY
    dsize.push_back(JSONNode("bsy", BLOCKSIZEY));
#endif
#ifdef CACHE_MODE
    dsize.push_back(JSONNode("mode", "cache_mode"));
#elif FLAT_MODE
    dsize.push_back(JSONNode("mode", "flat_mode"));
#endif

    auto get_float_bw = [&](std::string const& s, int touched_elements) -> double {
        auto size = sizeof(float) * touched_elements;
        auto time = times.min(s);
        return size/time/(1024.*1024.*1024.);
    };

    auto get_double_bw = [&](std::string const& s, int touched_elements) -> double {
        auto size = sizeof(double) * touched_elements;
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
      push_stencil_info(stencils, "copy", get_float_bw("copy", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "copyi1", get_float_bw("copyi1", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumi1", get_float_bw("sumi1", 2 * (isize - 2 * h + 1) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumj1", get_float_bw("sumj1", 2 * (isize - 2 * h) * (jsize - 2 * h + 1) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumk1", get_float_bw("sumk1", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h + 1)));
      push_stencil_info(stencils, "avgi", get_float_bw("avgi", 2 * (isize - 2 * h + 2) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "avgj", get_float_bw("avgj", 2 * (isize - 2 * h) * (jsize - 2 * h + 2) * (ksize - 2 * h)));
      push_stencil_info(stencils, "avgk", get_float_bw("avgk", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h + 2)));
      push_stencil_info(stencils, "lap", get_float_bw("lap", 2 * (isize - 2 * h + 2) * (jsize - 2 * h + 2) * (ksize - 2 * h)));
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
      push_stencil_info(stencils, "copy", get_double_bw("copy", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "copyi1", get_double_bw("copyi1", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumi1", get_double_bw("sumi1", 2 * (isize - 2 * h + 1) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumj1", get_double_bw("sumj1", 2 * (isize - 2 * h) * (jsize - 2 * h + 1) * (ksize - 2 * h)));
      push_stencil_info(stencils, "sumk1", get_double_bw("sumk1", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h + 1)));
      push_stencil_info(stencils, "avgi", get_double_bw("avgi", 2 * (isize - 2 * h + 2) * (jsize - 2 * h) * (ksize - 2 * h)));
      push_stencil_info(stencils, "avgj", get_double_bw("avgj", 2 * (isize - 2 * h) * (jsize - 2 * h + 2) * (ksize - 2 * h)));
      push_stencil_info(stencils, "avgk", get_double_bw("avgk", 2 * (isize - 2 * h) * (jsize - 2 * h) * (ksize - 2 * h + 2)));
      push_stencil_info(stencils, "lap", get_double_bw("lap", 2 * (isize - 2 * h + 2) * (jsize - 2 * h + 2) * (ksize - 2 * h)));
      precd.push_back(stencils);
    }

    dsize.push_back(precd);

    globalNode.push_back(dsize);

    if(write_out) {
        std::ofstream fs(out_name);
        fs << globalNode.write_formatted() << std::endl;
    }

    return 0;
}
