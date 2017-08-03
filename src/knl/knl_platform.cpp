#include "knl/knl_platform.h"

namespace platform {

namespace knl {

void flat::setup(arguments& args) {
  arguments& pargs = args.command(name, "variant");
  arguments& vargs_1d = pargs.command("1d");
}

variant_base* flat::create_variant(const arguments_map& args) {
  if (args.get("platform") != name) return nullptr;

  if (args.get("precision") == "single") {
    if (args.get("variant") == "1d") return new variant_1d<flat, float>(args);
  } else if (args.get("precision") == "double") {
    if (args.get("variant") == "1d") return new variant_1d<flat, double>(args);
  }

  return nullptr;
}

void cache::setup(arguments& args) {
  arguments& pargs = args.command(name, "variant");
  arguments& vargs_1d = pargs.command("1d");
}

variant_base* cache::create_variant(const arguments_map& args) {
  if (args.get("platform") != name) return nullptr;

  if (args.get("precision") == "single") {
    if (args.get("variant") == "1d") return new variant_1d<cache, float>(args);
  } else if (args.get("precision") == "double") {
    if (args.get("variant") == "1d") return new variant_1d<cache, double>(args);
  }

  return nullptr;
}

}  // namespace knl

}  // namespace platform
