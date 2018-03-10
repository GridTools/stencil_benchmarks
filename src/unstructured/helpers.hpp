#pragma once

__host__ __device__ inline unsigned int
uindex(const unsigned int i, const unsigned int c, const unsigned int j,
       const unsigned int cstride, const unsigned int jstride,
       const unsigned int first_padding) {
  return i + c * cstride + j * jstride + first_padding;
}

__host__ __device__ inline unsigned int
uindex3(const unsigned int i, const unsigned int c, const unsigned int j,
        const unsigned int k, const unsigned int cstride,
        const unsigned int jstride, const unsigned int kstride,
        const unsigned int first_padding) {
  return i + c * cstride + j * jstride + k * kstride + first_padding;
}
