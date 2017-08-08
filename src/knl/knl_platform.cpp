#include "knl/knl_platform.h"

#include "knl/knl_variant_1d.h"
#include "knl/knl_variant_1d_nontemporal.h"

namespace platform {

    namespace knl {

        void flat::setup(arguments &args) {
            arguments &pargs = args.command(name, "variant");
            pargs.command("1d");
            pargs.command("1d-nontemporal");
        }

        variant_base *flat::create_variant(const arguments_map &args) {
            if (args.get("platform") != name)
                return nullptr;

            std::string prec = args.get("precision");
            std::string var = args.get("variant");

            if (prec == "single") {
                if (var == "1d")
                    return new variant_1d< flat, float >(args);
                if (var == "1d-nontemporal")
                    return new variant_1d_nontemporal< flat, float >(args);
            } else if (prec == "double") {
                if (var == "1d")
                    return new variant_1d< flat, double >(args);
                if (var == "1d-nontemporal")
                    return new variant_1d_nontemporal< flat, double >(args);
            }

            return nullptr;
        }

        void cache::setup(arguments &args) {
            arguments &pargs = args.command(name, "variant");
            pargs.command("1d");
            pargs.command("1d-nontemporal");
        }

        variant_base *cache::create_variant(const arguments_map &args) {
            if (args.get("platform") != name)
                return nullptr;

            std::string prec = args.get("precision");
            std::string var = args.get("variant");

            if (prec == "single") {
                if (var == "1d")
                    return new variant_1d< cache, float >(args);
                if (var == "1d-nontemporal")
                    return new variant_1d_nontemporal< cache, float >(args);
            } else if (prec == "double") {
                if (var == "1d")
                    return new variant_1d< cache, double >(args);
                if (var == "1d-nontemporal")
                    return new variant_1d_nontemporal< cache, double >(args);
            }

            return nullptr;
        }

    } // namespace knl

} // namespace platform
