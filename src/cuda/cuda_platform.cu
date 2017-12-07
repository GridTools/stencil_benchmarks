#include "cuda/cuda_platform.h"

#include "cuda/cuda_vadv_variant.h"
#include "cuda/cuda_hdiff_variant.h"
#include "cuda/cuda_hdiff_variant_noshared.h"
#include "cuda/cuda_variant_ij_blocked.h"

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
            pargs.command("hdiff")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            pargs.command("hdiff-noshared")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8")
                .add("k-blocksize", "block size in k-direction", "8");
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
                    return new vadv_variant<cuda, float>(args);
                if (var == "hdiff")
                    return new hdiff_variant<cuda, float>(args);
                if (var == "hdiff-noshared")
                    return new hdiff_variant_noshared<cuda, float>(args);
            } else if (prec == "double") {
                if (var == "ij-blocked")
                    return new variant_ij_blocked<cuda, double>(args);
                if (var == "vadv")
                    return new vadv_variant<cuda, double>(args);
                if (var == "hdiff")
                    return new hdiff_variant<cuda, double>(args);
                if (var == "hdiff-noshared")
                    return new hdiff_variant_noshared<cuda, double>(args);
            }

            return nullptr;
        }

        void cuda::limit_blocksize(int &iblocksize, int &jblocksize) {
            int kblocksize = 1;
            limit_blocksize(iblocksize, jblocksize, kblocksize);
        }

        void cuda::limit_blocksize(int &iblocksize, int &jblocksize, int &kblocksize) {
            if (iblocksize <= 0 || jblocksize <= 0 || kblocksize <= 0)
                throw ERROR("invalid CUDA block size");

            cudaError_t err;
            int device;
            if ((err = cudaGetDevice(&device)) != cudaSuccess)
                throw ERROR("error in cudaGetDevice: " + std::string(cudaGetErrorString(err)));
            cudaDeviceProp prop;
            if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess)
                throw ERROR("error in cudaGetDeviceProperties: " + std::string(cudaGetErrorString(err)));

            int iblocksize0 = iblocksize, jblocksize0 = jblocksize, kblocksize0 = kblocksize;
            bool adapt = false;
            if (iblocksize > prop.maxThreadsDim[0]) {
                iblocksize = prop.maxThreadsDim[0];
                adapt = true;
            }
            if (jblocksize > prop.maxThreadsDim[1]) {
                jblocksize = prop.maxThreadsDim[1];
                adapt = true;
            }
            if (kblocksize > prop.maxThreadsDim[2]) {
                kblocksize = prop.maxThreadsDim[2];
                adapt = true;
            }

            while (iblocksize * jblocksize * kblocksize > prop.maxThreadsPerBlock) {
                if (iblocksize > jblocksize) {
                    if (iblocksize > kblocksize)
                        iblocksize /= 2;
                    else
                        kblocksize /= 2;
                } else {
                    if (jblocksize > kblocksize)
                        jblocksize /= 2;
                    else
                        kblocksize /= 2;
                }
                adapt = true;
            }
            if (adapt) {
                std::cerr << "WARNING: adapted CUDA block size to conform to device limits "
                          << "(" << iblocksize0 << "x" << jblocksize0 << "x" << kblocksize0 << " to " << iblocksize
                          << "x" << jblocksize << "x" << kblocksize << ")" << std::endl;
            }

            if (iblocksize <= 0 || jblocksize <= 0 || kblocksize <= 0)
                throw ERROR("CUDA block size adaption failed");
        }

    } // namespace cuda

} // namespace platform
