#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <omp.h>

#include "arguments.h"
#include "bandwidth_counter.h"
#include "counter.h"
#include "except.h"
#include "papi_counter.h"
#include "platform.h"
#include "table.h"
#include "timer.h"
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

std::string metric_name(const arguments_map &args) {
    auto m = args.get("metric");
    auto mk = args.get("metric-kind");
    auto ma = args.get("metric-accumulate");
    if (ma == "none")
        return m + "-" + mk;
    else
        return m + "-" + mk + "-" + ma;
}

struct result {
    std::string stencil;
    std::vector<double> data;
};

std::vector<result> run_stencils(const arguments_map &args, bool per_thread = false) {
    std::vector<result> results;

    auto variant = platform::create_variant(args);
    auto stencil = args.get("stencil");
    auto m = args.get("metric");
    auto mk = args.get("metric-kind");
    auto ma = args.get("metric-accumulate");

    std::vector<std::string> stencils;
    if (stencil == "all")
        stencils = variant->stencil_list();
    else
        stencils = {stencil};

    for (const auto &s : stencils) {
        std::unique_ptr<counter> ctr;
        if (m == "time")
            ctr.reset(new timer());
        else if (m == "bandwidth")
            ctr.reset(new bandwidth_counter(variant->touched_bytes(s)));
        else if (m == "papi")
            ctr.reset(new papi_counter(args.get("papi-event")));
        else
            throw ERROR("invalid metric");
        variant->run(s, *ctr);

        result r;
        r.stencil = s;
        if (per_thread && mk == "total") {
            for (int t = 0; t < ctr->threads(); ++t) {
                result_array ra = ctr->thread_total(t);
                if (ma == "avg")
                    r.data.push_back(ra.avg());
                else if (ma == "min")
                    r.data.push_back(ra.min());
                else if (ma == "max")
                    r.data.push_back(ra.max());
                else
                    throw ERROR("invalid value for metric-accumulate");
            }
        } else {
            result_array ra;
            if (mk == "total")
                ra = ctr->total();
            else if (mk == "imbalance")
                ra = ctr->imbalance();
            else
                throw ERROR("invalid metric-kind");
            if (ma == "avg")
                r.data.push_back(ra.avg());
            else if (ma == "min")
                r.data.push_back(ra.min());
            else if (ma == "max")
                r.data.push_back(ra.max());
            else
                throw ERROR("invalid value for metric-accumulate");
        }
        results.push_back(r);
    }

    return results;
}

void run_single_size(const arguments_map &args, std::ostream &out) {
    out << "# times are given in milliseconds, bandwidth in GB/s" << std::endl;

    table t(2);
    t << "stencil" << metric_name(args);

    std::string m = metric_name(args);
    const auto res = run_stencils(args, true);

    if (res.front().data.size() > 1) {
        table t(res.front().data.size() + 1);
        t << "stencil";
        for (int i = 0; i < res.front().data.size(); ++i)
            t << (m + "-" + std::to_string(i));
        for (auto &r : res) {
            t << r.stencil;
            for (auto &v : r.data)
                t << v;
        }
        out << t;
    } else {
        table t(2);
        t << "stencil" << m;
        for (auto &r : res)
            t << r.stencil << r.data.front();
        out << t;
    }
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
    const int halo = args.get<int>("halo");
    for (int size = min_size; size <= isize_max + 2 * halo; size *= 2) {
        std::stringstream size_stream;
        size_stream << (size - 2 * halo);

        auto res = run_stencils(args.with({{"i-size", size_stream.str()}, {"j-size", size_stream.str()}}));
        for (auto &r : res) {
            res_map[r.stencil].push_back(r.data.front());
        }
        ++sizes;
    }

    table t(sizes + 1);
    t << "Stencil";
    for (int size = min_size; size <= isize_max + 2 * halo; size *= 2)
        t << (size - 2 * halo);

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
            t << res.front().data.front();
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
        .add("metric", "what to measure (time, bandwidth, papi)", "bandwidth")
        .add("metric-kind", "kind of measurement (total, imbalance)", "total")
        .add("metric-accumulate", "accumulation of result over runs (avg, min, max)", "avg")
        .add("papi-event", "PAPI event name", "PAPI_L2_TCM")
        .add("output", "output file", "stdout")
        .add("runs", "number of runs", "20")
        .add_flag("no-header", "do not print header");

    platform::setup(args);

    auto argsmap = args.parse(argc, argv);

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
        throw ERROR("invalid run-mode");

    return 0;
}
