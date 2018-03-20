#pragma once

#ifdef __GNUC__
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GT_FORCE_INLINE inline __forceinline
#else
#define GT_FORCE_INLINE inline
#endif

#ifndef GT_FUNCTION
#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#define GT_FUNCTION_HOST __host__ __forceinline__
#define GT_FUNCTION_DEVICE __device__ __forceinline__
#define GT_FUNCTION_WARNING __host__ __device__
#else
#define GT_FUNCTION GT_FORCE_INLINE
#define GT_FUNCTION_HOST GT_FORCE_INLINE
#define GT_FUNCTION_DEVICE GT_FORCE_INLINE
#define GT_FUNCTION_WARNING
#endif
#endif

enum class location { cell = 0, edge, vertex };

GT_FUNCTION
constexpr unsigned int num_colors(location loc) { return loc == location::cell ? 2 : (loc == location::edge ? 3 : 1); }

GT_FUNCTION
static constexpr size_t num_nodes(const unsigned int isize, const unsigned int jsize, const unsigned int nhalo) {
    return (isize + nhalo * 2) * (jsize + nhalo * 2);
}

GT_FUNCTION
static constexpr size_t num_neighbours(const location primary_loc, const location neigh_loc) {
    return primary_loc == location::cell ? 3 : (primary_loc == location::edge ? (neigh_loc == location::edge ? 4 : 2)
                                                                              : (6));
}
