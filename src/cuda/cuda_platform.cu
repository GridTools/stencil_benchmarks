#include "cuda/cuda_platform.h"

#include "cuda/cuda_hdiff_variant.h"
#include "cuda/cuda_hdiff_variant_ijnoshared.h"
#include "cuda/cuda_hdiff_variant_incache.h"
#include "cuda/cuda_hdiff_variant_noshared.h"
#include "cuda/cuda_vadv_variant.h"
#include "cuda/cuda_variant_1d.h"
#include "cuda/cuda_variant_ij_blocked.h"
#include "cuda/cuda_variant_ijk_blocked.h"

namespace platform {

    namespace cuda {

        void cuda::setup(arguments &args) {
            auto &basic = args.command("basic", "variant");
            basic.command("1d").add("blocksize", "1D block size", "32");
            basic.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            basic.command("ijk-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8")
                .add("k-blocksize", "block size in k-direction", "8");
            auto &hdiff = args.command("hdiff", "variant");
            hdiff.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-noshared")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ijk-blocked-noshared")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8")
                .add("k-blocksize", "block size in k-direction", "8");
            hdiff.command("ijk-blocked-incache")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8")
                .add("k-blocksize", "block size in k-direction", "1");
            auto &vadv = args.command("vadv", "variant");
            vadv.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
        }

        namespace {
            template <class ValueType>
            variant_base *create_variant_by_prec(const arguments_map &args) {
                std::string grp = args.get("group");
                std::string var = args.get("variant");

                if (grp == "basic") {
                    if (var == "1d")
                        return new variant_1d<ValueType>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<ValueType>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<ValueType>(args);
                }
                if (grp == "hdiff") {
                    if (var == "ij-blocked")
                        return new hdiff_variant<ValueType>(args);
                    if (var == "ij-blocked-noshared")
                        return new hdiff_variant_ijnoshared<ValueType>(args);
                    if (var == "ijk-blocked-noshared")
                        return new hdiff_variant_noshared<ValueType>(args);
                    if (var == "ijk-blocked-incache")
                        return new hdiff_variant_incache<ValueType>(args);
                }
                if (grp == "vadv") {
                    if (var == "ij-blocked")
                        return new vadv_variant<ValueType>(args);
                }
                return nullptr;
            }
        }

        variant_base *cuda::create_variant(const arguments_map &args) {
            std::string prec = args.get("precision");

            if (prec == "single") {
                return create_variant_by_prec<float>(args);
            } else if (prec == "double") {
                return create_variant_by_prec<double>(args);
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
