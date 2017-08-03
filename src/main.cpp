#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

#include "arguments.h"
#include "except.h"
#include "platform.h"
#include "table.h"
#include "variant_base.h"

template <class Type>
std::string str(const Type& t) {
  std::stringstream s;
  s << t;
  return s.str();
}

std::vector<std::pair<std::string, result>> run_stencils(
    const arguments_map& args) {
  auto variant = platform::create_variant(args);
  auto stencil = args.get("stencil");
  std::vector<std::pair<std::string, result>> res;
  if (stencil == "all") {
    for (auto& s : variant->stencil_list())
      res.emplace_back(s, variant->run(s));
  } else {
    res.emplace_back(stencil, variant->run(stencil));
  }
  return res;
}

void run_single_size(const arguments_map& args, std::ostream& out) {
  out << "# times are given in milliseconds, bandwidth in GB/s" << std::endl;

  table t(out, 7);
  t << "Stencil"
    << "Time-avg"
    << "Time-min"
    << "Time-max"
    << "BW-avg"
    << "BW-min"
    << "BW-max";

  auto print_result = [&t](const std::string s, const result& r) {
    t << s << (r.time.avg() * 1000) << (r.time.min() * 1000)
      << (r.time.max() * 1000) << r.bandwidth.avg() << r.bandwidth.min()
      << r.bandwidth.max();
  };

  const auto res = run_stencils(args);
  for (auto& r : res) print_result(r.first, r.second);
}

void run_ij_scaling(const arguments_map& args, std::ostream& out) {
  out << "# shown is the estimated max. bandwidth in GB/s" << std::endl;

  const int isize_max = args.get<int>("i-size");
  const int jsize_max = args.get<int>("j-size");
  if (isize_max != jsize_max)
    throw ERROR("i-size and j-size must be equal for ij-scaling mode");

  const auto stencils = platform::variant_base::stencil_list();

  std::string stencil = args.get("stencil");
  std::map<std::string, std::vector<double>> res_map;

  int sizes = 0;
  for (int size = 32; size <= isize_max; size *= 2) {
    auto res =
        run_stencils(args.with({{"i-size", str(size)}, {"j-size", str(size)}}));
    for (auto& r : res) res_map[r.first].push_back(r.second.bandwidth.max());
    ++sizes;
  }

  table t(out, sizes + 1);
  t << "Stencil";
  for (int size = 32; size <= isize_max; size *= 2) t << size;

  if (stencil == "all") {
    for (auto& s : stencils) {
      t << s;
      for (auto& r : res_map[s]) t << r;
    }
  } else {
    t << stencil;
    for (auto& r : res_map[stencil]) t << r;
  }
}

void print_header(const arguments_map& args, std::ostream& out) {
  std::size_t max_name_width = 0, max_value_width = 0;
  for (auto& a : args) {
    max_name_width = std::max(max_name_width, a.first.size());
    max_value_width = std::max(max_value_width, a.second.size());
  }

  int i = 0;
  for (auto& a : args) {
    if (i == 0) out << "# ";
    out << std::setw(max_name_width + 2) << std::right << (a.first + ": ")
        << std::setw(max_value_width) << std::left << a.second << "   ";
    if (++i >= 5) {
      out << std::endl;
      i = 0;
    }
  }
  if (i != 0) out << std::endl;
  out << "# ---" << std::endl;
}

int main(int argc, char** argv) {
  arguments args(argv[0], "platform");

  args.add("i-size", "domain size in i-direction", "1024")
      .add("j-size", "domain size in j-direction", "1024")
      .add("k-size", "domain size in k-direction", "80")
      .add("i-layout", "layout specifier", "2")
      .add("j-layout", "layout specifier", "1")
      .add("k-layout", "layout specifier", "0")
      .add("halo", "halo size", "2")
      .add("padding", "padding in elements", "1")
      .add("precision", "single or double precision", "double")
      .add("stencil", "stencil to run", "all")
      .add("run-mode", "run mode (single-size, ij-scaling)", "single-size")
      .add("output", "output file", "stdout")
      .add_flag("no-header", "do not print header");

  platform::setup(args);

  auto argsmap = args.parse(argc, argv);

  std::streambuf* buf;
  std::ofstream outfile;
  if (argsmap.get("output") == "stdout") {
    buf = std::cout.rdbuf();
  } else {
    outfile.open(argsmap.get("output"));
    buf = outfile.rdbuf();
  }
  std::ostream out(buf);

  if (!argsmap.get_flag("no-header")) print_header(argsmap, out);

  std::string run_mode = argsmap.get("run-mode");
  if (run_mode == "single-size")
    run_single_size(argsmap, out);
  else if (run_mode == "ij-scaling")
    run_ij_scaling(argsmap, out);

  return 0;
}
