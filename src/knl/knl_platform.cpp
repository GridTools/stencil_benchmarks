#include "knl/knl_platform.h"

#include "knl/knl_variant_1d.h"
#include "knl/knl_variant_1d_nontemporal.h"
#include "knl/knl_variant_ij_blocked.h"
#include "knl/knl_variant_ijk_blocked.h"

namespace platform {

    namespace knl {

        namespace {
            template <class Platform>
            void common_setup(arguments &args) {
                arguments &pargs = args.command(Platform::name, "variant");
                pargs.command("1d");
                pargs.command("1d-nontemporal");
                pargs.command("ij-blocked")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
                pargs.command("ijk-blocked")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8")
                    .add("k-blocksize", "block size in k-direction", "8");
            }

            template <class Platform>
            variant_base *common_create_variant(const arguments_map &args) {
                if (args.get("platform") != Platform::name)
                    return nullptr;

                std::string prec = args.get("precision");
                std::string var = args.get("variant");

                if (prec == "single") {
                    if (var == "1d")
                        return new variant_1d<Platform, float>(args);
                    if (var == "1d-nontemporal")
                        return new variant_1d_nontemporal<Platform, float>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<Platform, float>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<Platform, float>(args);
                } else if (prec == "double") {
                    if (var == "1d")
                        return new variant_1d<Platform, double>(args);
                    if (var == "1d-nontemporal")
                        return new variant_1d_nontemporal<Platform, double>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<Platform, double>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<Platform, double>(args);
                }

                return nullptr;
            }
        }

        void flat::setup(arguments &args) { common_setup<flat>(args); }

        variant_base *flat::create_variant(const arguments_map &args) { return common_create_variant<flat>(args); }

        void cache::setup(arguments &args) { common_setup<cache>(args); }

        variant_base *cache::create_variant(const arguments_map &args) { return common_create_variant<cache>(args); }

    } // namespace knl

} // namespace platform
