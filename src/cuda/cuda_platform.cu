#include "cuda/cuda_platform.h"

#include "cuda/cuda_variant_ij_blocked.h"
#include "cuda/cuda_variant_vadv.h"

namespace platform {

    namespace cuda {

        void cuda::setup(arguments &args) {
            arguments &pargs = args.command(name, "variant");
            pargs.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            pargs.command("vadv")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
        }

        variant_base *cuda::create_variant(const arguments_map &args) {
            if (args.get("platform") != name)
                return nullptr;

            std::string prec = args.get("precision");
            std::string var = args.get("variant");

            if (prec == "single") {
                if (var == "ij-blocked")
                    return new variant_ij_blocked<cuda, float>(args);
                if (var == "vadv")
                    return new variant_vadv<cuda, float>(args);
            } else if (prec == "double") {
                if (var == "ij-blocked")
                    return new variant_ij_blocked<cuda, double>(args);
                if (var == "vadv")
                    return new variant_vadv<cuda, double>(args);
            }

            return nullptr;
        }

        void cuda::limit_blocksize(int &iblocksize, int &jblocksize) {
            cudaError_t err;
            int device;
            if ((err = cudaGetDevice(&device)) != cudaSuccess)
                throw ERROR("error in cudaGetDevice: " + std::string(cudaGetErrorString(err)));
            cudaDeviceProp prop;
            if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess)
                throw ERROR("error in cudaGetDeviceProperties: " + std::string(cudaGetErrorString(err)));

            int iblocksize0 = iblocksize, jblocksize0 = jblocksize;
            bool adapt = false;
            if (iblocksize > prop.maxThreadsDim[0]) {
                iblocksize = prop.maxThreadsDim[0];
                adapt = true;
            }
            if (jblocksize > prop.maxThreadsDim[1]) {
                jblocksize = prop.maxThreadsDim[1];
                adapt = true;
            }

            while (iblocksize * jblocksize > prop.maxThreadsPerBlock) {
                if (iblocksize > jblocksize)
                    iblocksize /= 2;
                else
                    jblocksize /= 2;
                adapt = true;
            }
            if (adapt) {
                std::cerr << "WARNING: adapted CUDA block size to conform to device limits "
                          << "(" << iblocksize0 << "x" << jblocksize0 << " to " << iblocksize << "x" << jblocksize
                          << ")" << std::endl;
            }

            if (iblocksize <= 0 || jblocksize <= 0)
                throw ERROR("CUDA block size adaption failed");
        }

    } // namespace cuda

} // namespace platform
