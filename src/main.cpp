#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <omp.h>

#include "arguments.h"
#include "except.h"
#include "platform.h"
#include "table.h"
#include "variant_base.h"

void print_header(const arguments_map &args, std::ostream &out) {
    out << "# passed arguments:" << std::endl;

    const int cols = 3;
    table t(2 * cols + 1);
    int col = 0;
    for (auto &a : args) {
        if (col++ % cols == 0)
            t << "#";
        t << (a.first + ":") << a.second;
    }

    out << t;
}

std::string metric_info(const arguments_map &args) {
    std::string m = args.get("metric");
    if (m == "time")
        return "# shown is the measured min. time in ms";
    else if (m == "bandwidth")
        return "# shown is the estimated max. bandwidth in GB/s";
    else if (m == "papi")
        return "# shown is the measured min. counter value";
    else if (m == "papi-imbalance")
        return "# shown is the measured min. counter thread imbalance";
    else
        throw ERROR("invalid metric");
}

double get_metric(const arguments_map &args, const result &r) {
    std::string m = args.get("metric");
    if (m == "time")
        return r.time.min() * 1000;
    else if (m == "bandwidth")
        return r.bandwidth.max();
#ifdef WITH_PAPI
    else if (m == "papi")
        return r.counter.min();
    else if (m == "papi-imbalance")
        return r.counter_imbalance.min();
#endif
    else
        throw ERROR("invalid metric");
}

std::vector<result> run_stencils(const arguments_map &args) {
    auto variant = platform::device::create_variant(args);
    auto stencil = args.get("stencil");
    return variant->run(stencil);
}

void run_single_size(const arguments_map &args, std::ostream &out) {
    out << "# times are given in milliseconds, bandwidth in GB/s" << std::endl;

#ifdef WITH_PAPI
    table t(13);
#else
    table t(7);
#endif
    t << "Stencil"
      << "Time-avg"
      << "Time-min"
      << "Time-max"
      << "BW-avg"
      << "BW-min"
      << "BW-max"
#ifdef WITH_PAPI
      << "CTR-avg"
      << "CTR-min"
      << "CTR-max"
      << "CTR-IMB-avg"
      << "CTR-IMB-min"
      << "CTR-IMB-max"
#endif
        ;

    auto print_result = [&t](const result &r) {
        t << r.stencil << (r.time.avg() * 1000) << (r.time.min() * 1000) << (r.time.max() * 1000) << r.bandwidth.avg()
          << r.bandwidth.min() << r.bandwidth.max()
#ifdef WITH_PAPI
          << r.counter.avg() << r.counter.min() << r.counter.max() << r.counter_imbalance.avg()
          << r.counter_imbalance.min() << r.counter_imbalance.max()
#endif
            ;
    };

    const auto res = run_stencils(args);
    for (auto &r : res)
        print_result(r);

    out << t;
}

void run_tabular(const arguments_map &args, std::ostream &out) {
    std::string stencil = args.get("stencil");

    int isize = args.get<int>("i-size");
    int jsize = args.get<int>("j-size");
    int ksize = args.get<int>("k-size");
    int iblocksize = isize;
    int jblocksize = jsize;
    int kblocksize = ksize;
    if (args.exists("i-blocksize"))
        iblocksize = args.get<int>("i-blocksize");
    if (args.exists("j-blocksize"))
        jblocksize = args.get<int>("j-blocksize");
    if (args.exists("k-blocksize"))
        kblocksize = args.get<int>("k-blocksize");

    if (iblocksize > isize || jblocksize > jsize || kblocksize > ksize) {
        std::cerr << "Warning: invalid block size configuration, no computation performed" << std::endl;
        return;
    }
    int threads = args.get<int>("threads");
    int iblocks = (isize + iblocksize - 1) / iblocksize;
    int jblocks = (jsize + jblocksize - 1) / jblocksize;
    int kblocks = (ksize + kblocksize - 1) / kblocksize;
    if (iblocks * jblocks * kblocks < threads) {
        std::cerr << "Warning: not enough blocks for the given threads, no computation performed" << std::endl;
        return;
    }

#ifdef WITH_PAPI
    table t(args.size() + 13);
#else
    table t(args.size() + 7);
#endif

    for (auto &arg : args)
        t << arg.first;

    t << "stencil"
      << "time-avg"
      << "time-min"
      << "time-max"
      << "bandwidth-avg"
      << "bandwidth-min"
      << "bandwidth-max"
#ifdef WITH_PAPI
      << "counter-avg"
      << "counter-min"
      << "counter-max"
      << "counter-imbalance-avg"
      << "counter-imbalance-min"
      << "counter-imbalance-max"
#endif
        ;

    const auto res = run_stencils(args);

    for (auto &r : res) {
        for (auto &arg : args)
            t << arg.second;

        t << r.stencil << (r.time.avg() * 1000) << (r.time.min() * 1000) << (r.time.max() * 1000) << r.bandwidth.avg()
          << r.bandwidth.min() << r.bandwidth.max()
#ifdef WITH_PAPI
          << r.counter.avg() << r.counter.min() << r.counter.max() << r.counter_imbalance.avg()
          << r.counter_imbalance.min() << r.counter_imbalance.max()
#endif
            ;
    }

    out << t;
}

void run_ij_scaling(const arguments_map &args, std::ostream &out) {
    out << metric_info(args) << std::endl;

    const int isize_max = args.get<int>("i-size");
    const int jsize_max = args.get<int>("j-size");
    const int min_size = args.get<int>("min-size");
    if (isize_max != jsize_max)
        throw ERROR("i-size and j-size must be equal for ij-scaling mode");
    if (min_size <= 0)
        throw ERROR("invalid min-size < 1");

    std::string stencil = args.get("stencil");
    std::map<std::string, std::vector<double>> res_map;

    int sizes = 0;
    for (int size = min_size; size <= isize_max; size *= 2) {
        std::stringstream size_stream;
        size_stream << size;

        auto res = run_stencils(args.with({{"i-size", size_stream.str()}, {"j-size", size_stream.str()}}));
        for (auto &r : res) {
            res_map[r.stencil].push_back(get_metric(args, r));
        }
        ++sizes;
    }

    table t(sizes + 1);
    t << "Stencil";
    for (int size = min_size; size <= isize_max; size *= 2)
        t << size;

    std::set<std::string> stencils;
    for (const auto &r : res_map)
        stencils.insert(r.first);

    if (stencil == "all") {
        for (const auto &s : stencils) {
            t << s;
            for (auto &r : res_map[s])
                t << r;
        }
    } else {
        t << stencil;
        for (const auto &r : res_map[stencil])
            t << r;
    }
    out << t;
}

void run_blocksize_scan(const arguments_map &args, std::ostream &out) {
    out << metric_info(args) << std::endl;

    std::string stencil = args.get("stencil");
    if (stencil == "all")
        throw ERROR("blocksize-scan run-mode can only be used with a single stencil");

    const int isize = args.get<int>("i-size");
    const int jsize = args.get<int>("j-size");
    const int min_size = args.get<int>("min-size");
    if (min_size <= 0)
        throw ERROR("invalid min-size < 1");

    int jsizes = 0;
    for (int jblocksize = min_size; jblocksize < 2 * jsize; jblocksize *= 2)
        ++jsizes;
    table t(jsizes + 1);

    t << "i\\j";
    for (int jblocksize = min_size; jblocksize < 2 * jsize; jblocksize *= 2)
        t << jblocksize;

    for (int iblocksize = min_size; iblocksize < 2 * isize; iblocksize *= 2) {
        t << iblocksize;
        std::stringstream ibs;
        ibs << iblocksize;
        for (int jblocksize = min_size; jblocksize < 2 * jsize; jblocksize *= 2) {
            std::stringstream jbs;
            jbs << jblocksize;
            auto res = run_stencils(args.with({{"i-blocksize", ibs.str()}, {"j-blocksize", jbs.str()}}));
            t << get_metric(args, res.front());
        }
    }
    out << t;
}

void run_block_scan(const arguments_map &args, std::ostream &out) {
    std::string stencil = args.get("stencil");
    if (stencil == "all")
        throw ERROR("block-scan run-mode can only be used with a single stencil");

    const int isize = args.get<int>("i-size");
    const int jsize = args.get<int>("j-size");
    const int ksize = args.get<int>("k-size");
    const int threads = omp_get_max_threads();

    std::vector<std::array<int, 3>> blocks;

    for (int k = 1; k <= ksize; ++k) {
        for (int j = 1; j <= jsize; ++j) {
            for (int i = 1; i <= isize; ++i) {
                if (i * j * k == threads)
                    blocks.push_back({i, j, k});
            }
        }
    }

#ifdef WITH_PAPI
    table t(13);
#else
    table t(7);
#endif
    t << "Block-size"
      << "Time-avg"
      << "Time-min"
      << "Time-max"
      << "BW-avg"
      << "BW-min"
      << "BW-max"
#ifdef WITH_PAPI
      << "CTR-avg"
      << "CTR-min"
      << "CTR-max"
      << "CTR-IMB-avg"
      << "CTR-IMB-min"
      << "CTR-IMB-max"
#endif
        ;

    auto print_result = [&t](const arguments_map &a, const result &r) {
        t << (a.get("i-blocks") + "x" + a.get("j-blocks") + "x" + a.get("k-blocks"));
        t << (r.time.avg() * 1000) << (r.time.min() * 1000) << (r.time.max() * 1000) << r.bandwidth.avg()
          << r.bandwidth.min() << r.bandwidth.max()
#ifdef WITH_PAPI
          << r.counter.avg() << r.counter.min() << r.counter.max() << r.counter_imbalance.avg()
          << r.counter_imbalance.min() << r.counter_imbalance.max()
#endif
            ;
    };

    for (const auto &b : blocks) {
        auto a = args.with({{"i-blocks", std::to_string(b[0])},
            {"j-blocks", std::to_string(b[1])},
            {"k-blocks", std::to_string(b[2])}});
        auto r = run_stencils(a).front();
        print_result(a, r);
    }

    out << t;
}

int main(int argc, char **argv) {
    arguments args(argv[0], "group");

    args.add("i-size", "domain size in i-direction", "1024")
        .add("j-size", "domain size in j-direction", "1024")
        .add("k-size", "domain size in k-direction", "80")
        .add("i-layout", "layout specifier", "2")
        .add("j-layout", "layout specifier", "1")
        .add("k-layout", "layout specifier", "0")
        .add("min-size", "minimum size/block size in ij-scaling and blocksize-scan run-modes", "1")
        .add("halo", "halo size", "2")
        .add("alignment", "alignment in elements", "1")
        .add("precision", "single or double precision", "double")
        .add("stencil", "stencil to run", "all")
        .add("run-mode", "run mode (single-size, ij-scaling, blocksize-scan, block-scan, tabular)", "single-size")
        .add("threads", "number of threads to use (0 = use OMP_NUM_THREADS)", "0")
        .add("metric", "what to measure (time, bandwidth, papi, papi-imbalance)", "bandwidth")
#ifdef WITH_PAPI
        .add("papi-event", "PAPI event name", "PAPI_L2_TCM")
#endif
        .add("output", "output file", "stdout")
        .add("runs", "number of runs", "20")
        .add_flag("no-header", "do not print header");

    platform::device::setup(args);

    auto argsmap = args.parse(argc, argv);

    argsmap.add("platform", platform::device::name);

    std::streambuf *buf;
    std::ofstream outfile;
    if (argsmap.get("output") == "stdout") {
        buf = std::cout.rdbuf();
    } else {
        outfile.open(argsmap.get("output"));
        if (!outfile)
            throw ERROR("could not open file '" + argsmap.get("output") + "'");
        buf = outfile.rdbuf();
    }
    std::ostream out(buf);

    if (!argsmap.get_flag("no-header") && argsmap.get("run-mode") != "tabular")
        print_header(argsmap, out);

    omp_set_dynamic(0);
    if (int threads = argsmap.get<int>("threads"))
        omp_set_num_threads(threads);

    std::string run_mode = argsmap.get("run-mode");
    if (run_mode == "single-size")
        run_single_size(argsmap, out);
    else if (run_mode == "ij-scaling")
        run_ij_scaling(argsmap, out);
    else if (run_mode == "blocksize-scan")
        run_blocksize_scan(argsmap, out);
    else if (run_mode == "block-scan")
        run_block_scan(argsmap, out);
    else if (run_mode == "tabular")
        run_tabular(argsmap, out);
    else
        throw ERROR("invalid run-mode");

    return 0;
}
