#include <cstring>
#include <fstream>
#include <iostream>

#include "arguments.h"
#include "platform.h"

void run_single(const arguments_map& args) {
  auto variant = platform::create_variant(args);
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
    out << std::setw(max_name_width + 2) << (a.first + ": ")
        << std::setw(max_value_width) << a.second << "   ";
    if (++i >= 5) {
      out << std::endl;
      i = 0;
    }
  }
  if (i != 0) out << std::endl;
}

int main(int argc, char** argv) {
  arguments args(argv[0], "platform");

  args.add("i-size", "domain size in i-direction", "1024")
      .add("j-size", "domain size in j-direction", "1024")
      .add("k-size", "domain size in k-direction", "1024")
      .add("i-layout", "layout specifier", "2")
      .add("j-layout", "layout specifier", "1")
      .add("k-layout", "layout specifier", "0")
      .add("halo", "halo size", "2")
      .add("padding", "padding in elements", "1")
      .add("precision", "single or double precision", "double")
      .add("stencil", "stencil to run", "all")
      .add("run-mode", "run mode (single, full)", "single")
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

  auto variant = platform::create_variant(argsmap);

  out << variant->run("copy", 5);

  return 0;
}
