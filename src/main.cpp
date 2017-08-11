#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

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

std::vector<std::pair<std::string, result>> run_stencils(const arguments_map &args) {
    auto variant = platform::create_variant(args);
    auto stencil = args.get("stencil");
    std::vector<std::pair<std::string, result>> res;
    if (stencil == "all") {
        for (auto &s : variant->stencil_list())
            res.emplace_back(s, variant->run(s));
    } else {
        res.emplace_back(stencil, variant->run(stencil));
    }
    return res;
}

void run_single_size(const arguments_map &args, std::ostream &out) {
    out << "# times are given in milliseconds, bandwidth in GB/s" << std::endl;

    table t(7);
    t << "Stencil"
      << "Time-avg"
      << "Time-min"
      << "Time-max"
      << "BW-avg"
      << "BW-min"
      << "BW-max";

    auto print_result = [&t](const std::string s, const result &r) {
        t << s << (r.time.avg() * 1000) << (r.time.min() * 1000) << (r.time.max() * 1000) << r.bandwidth.avg()
          << r.bandwidth.min() << r.bandwidth.max();
    };

    const auto res = run_stencils(args);
    for (auto &r : res)
        print_result(r.first, r.second);

    out << t;
}

void run_ij_scaling(const arguments_map &args, std::ostream &out) {
    out << "# shown is the estimated max. bandwidth in GB/s" << std::endl;

    const int isize_max = args.get<int>("i-size");
    const int jsize_max = args.get<int>("j-size");
    const int min_size = args.get<int>("min-size");
    if (isize_max != jsize_max)
        throw ERROR("i-size and j-size must be equal for ij-scaling mode");
    if (min_size <= 0)
        throw ERROR("invalid min-size < 1");

    const auto stencils = platform::variant_base::stencil_list();

    std::string stencil = args.get("stencil");
    std::map<std::string, std::vector<double>> res_map;

    int sizes = 0;
    const int halo = args.get<int>("halo");
    for (int size = min_size; size <= isize_max + 2 * halo; size *= 2) {
        std::stringstream size_stream;
        size_stream << (size - 2 * halo);

        auto res = run_stencils(args.with({{"i-size", size_stream.str()}, {"j-size", size_stream.str()}}));
        for (auto &r : res)
            res_map[r.first].push_back(r.second.bandwidth.max());
        ++sizes;
    }

    table t(sizes + 1);
    t << "Stencil";
    for (int size = min_size; size <= isize_max + 2 * halo; size *= 2)
        t << (size - 2 * halo);

    if (stencil == "all") {
        for (auto &s : stencils) {
            t << s;
            for (auto &r : res_map[s])
                t << r;
        }
    } else {
        t << stencil;
        for (auto &r : res_map[stencil])
            t << r;
    }
    out << t;
}

void run_blocksize_scan(const arguments_map &args, std::ostream &out) {
    out << "# shown is the estimated max. bandwidth in GB/s" << std::endl;

    std::string stencil = args.get("stencil");
    if (stencil == "all")
        throw ERROR("blocksize-scan run-mode can only be used with a single stencil");

    const int isize = args.get<int>("i-size");
    const int jsize = args.get<int>("j-size");
    const int min_size = args.get<int>("min-size");
    if (min_size <= 0)
        throw ERROR("invalid min-size < 1");

    int jsizes = 0;
    for (int jblocksize = min_size; jblocksize <= jsize; jblocksize *= 2)
        ++jsizes;
    table t(jsizes + 1);

    t << "i\\j";
    for (int jblocksize = min_size; jblocksize <= jsize; jblocksize *= 2)
        t << jblocksize;

    for (int iblocksize = min_size; iblocksize <= isize; iblocksize *= 2) {
        t << iblocksize;
        std::stringstream ibs;
        ibs << iblocksize;
        for (int jblocksize = min_size; jblocksize <= jsize; jblocksize *= 2) {
            std::stringstream jbs;
            jbs << jblocksize;
            auto res = run_stencils(args.with({{"i-blocksize", ibs.str()}, {"j-blocksize", jbs.str()}}));
            t << res.front().second.bandwidth.max();
        }
    }
    out << t;
}

int main(int argc, char **argv) {
    arguments args(argv[0], "platform");

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
        .add("run-mode", "run mode (single-size, ij-scaling, blocksize-scan)", "single-size")
        .add("threads", "number of threads to use (0 = use OMP_NUM_THREADS)", "0")
        .add("output", "output file", "stdout")
        .add_flag("no-header", "do not print header");

    platform::setup(args);

    auto argsmap = args.parse(argc, argv);

    std::streambuf *buf;
    std::ofstream outfile;
    if (argsmap.get("output") == "stdout") {
        buf = std::cout.rdbuf();
    } else {
        outfile.open(argsmap.get("output"));
        buf = outfile.rdbuf();
    }
    std::ostream out(buf);

    if (!argsmap.get_flag("no-header"))
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
    else
        throw ERROR("unknown run-mode");

    return 0;
}
