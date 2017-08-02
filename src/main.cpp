#include <cstring>
#include <iostream>

#include "arguments.h"
#include "platform.h"

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
      .add("output", "output file", "stdout")
      .add_flag("print-args", "print all arguments");

  platform::setup(args);

  auto argsmap = args.parse(argc, argv);

  if (argsmap.get_flag("print-args")) std::cout << argsmap;

  auto variant = platform::create_variant(argsmap);

  std::cout << variant->run("copy", 5) << std::endl;

  return 0;
}
