/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include "array.hpp"
#include <sstream>
#include <iostream>
#include <fstream>
#include <array>
#include <cuda_runtime.h>
#include "udefs.hpp"

class ntable {
public:
  GT_FUNCTION
  ntable(ntable const &other)
      : m_ploc(other.m_ploc), m_nloc(other.m_nloc), m_size(other.m_size),
        m_data(other.m_data) {}

  ntable(location primary_loc, location neigh_loc, size_t size)
      : m_ploc(primary_loc), m_nloc(neigh_loc), m_size(size) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_data, m_size * sizeof(size_t));
#else
    m_data = (size_t *)malloc(m_size * sizeof(size_t));
#endif
  }

  GT_FUNCTION
  size_t size() const { return m_size; }
  GT_FUNCTION
  size_t &raw_data(const unsigned int idx) { return m_data[idx]; }

  GT_FUNCTION
  size_t *data() { return m_data; }

  GT_FUNCTION
  location ploc() const { return m_ploc; }
  GT_FUNCTION
  location nloc() const { return m_nloc; }

protected:
  location m_ploc;
  location m_nloc;
  size_t m_size;
  size_t *m_data;
};

class sneighbours_table : public ntable {
public:
  GT_FUNCTION
  static constexpr size_t size_of_array(const location primary_loc,
                                        const location neigh_loc,
                                        const unsigned int isize,
                                        const unsigned int jsize,
                                        const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo + 1) * num_colors(primary_loc) *
           sizeof(size_t) * num_neighbours(primary_loc, neigh_loc);
  }

  GT_FUNCTION
  sneighbours_table(sneighbours_table const &other)
      : m_isize(other.m_isize), m_jsize(other.m_jsize), m_nhalo(other.m_nhalo),
        ntable(other) {}

  sneighbours_table(location primary_loc, location neigh_loc, size_t isize,
                    size_t jsize, size_t nhalo)
      : m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        ntable(primary_loc, neigh_loc,
               num_nodes(isize, jsize, nhalo + 1) * num_colors(primary_loc) *
                   num_neighbours(primary_loc, neigh_loc)) {}

  GT_FUNCTION
  size_t &operator()(int i, unsigned int c, int j, unsigned int neigh_idx) {

    assert(index_in_tables(i, c, j, neigh_idx) <
           size_of_array(m_ploc, m_nloc, m_isize, m_jsize, m_nhalo));

    return m_data[index_in_tables(i, c, j, neigh_idx)];
  }

  GT_FUNCTION
  size_t &operator()(size_t idx, unsigned int neigh_idx) {

    return m_data[neigh_index(idx, neigh_idx)];
  }

  GT_FUNCTION
  size_t last_compute_domain_idx() {
    return num_neighbours(m_ploc, m_nloc) * (m_isize)*num_colors(m_ploc) *
           (m_jsize);
  }

  GT_FUNCTION
  size_t last_west_halo_idx() {
    return last_compute_domain_idx() +
           num_neighbours(m_ploc, m_nloc) * m_jsize * m_nhalo *
               num_colors(m_ploc);
  }

  GT_FUNCTION
  size_t last_south_halo_idx() {
    return last_west_halo_idx() +
           num_neighbours(m_ploc, m_nloc) * m_nhalo * (m_nhalo + m_isize) *
               num_colors(m_ploc);
  }

  GT_FUNCTION
  size_t last_east_halo_idx() {
    return last_south_halo_idx() +
           num_neighbours(m_ploc, m_nloc) * (m_nhalo + m_jsize) * m_nhalo *
               num_colors(m_ploc);
  }

  GT_FUNCTION
  size_t last_north_halo_idx() {
    return last_east_halo_idx() +
           num_neighbours(m_ploc, m_nloc) * m_nhalo * num_colors(m_ploc) *
               (m_isize + m_nhalo * 2);
  }

  GT_FUNCTION
  size_t neigh_index(size_t idx, unsigned int neigh_idx) {
    assert(idx < last_compute_domain_idx());

    return idx + neigh_idx * (m_isize)*num_colors(m_ploc) * (m_jsize);
  }

  GT_FUNCTION
  size_t index_in_tables(int i, unsigned int c, int j, unsigned int neigh_idx) {
    assert(i >= -(int)m_nhalo && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(m_ploc));
    assert(j >= -(int)m_nhalo && j < (int)(m_jsize + m_nhalo));
    assert(neigh_idx < num_neighbours(m_ploc, m_nloc));

    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize) {
      int idx = i + c * m_isize + j * num_colors(m_ploc) * m_isize +
                neigh_idx * (m_isize)*num_colors(m_ploc) * (m_jsize);
      /*      if (idx >= last_compute_domain_idx()) {
              std::cout << "WARNING IN COMPUTE DOMAIN : " << i << "," << c <<
         "," << j
                        << "," << neigh_idx << ": " << last_compute_domain_idx()
                        << " -> " << idx << std::endl;
            }
      */
      return idx;
    }
    if (i < 0 && j >= 0 && j < (int)m_jsize) {
      int idx = (int)last_compute_domain_idx() - i - 1 + c * m_nhalo +
                j * m_nhalo * num_colors(m_ploc) +
                neigh_idx * m_jsize * m_nhalo * num_colors(m_ploc);
      /*      if (idx >= last_west_halo_idx())
              std::cout << "WARNING IN WEST : " << i << "," << c << "," << j <<
         ","
                        << neigh_idx << ": " << last_west_halo_idx() << " -> "
         << idx
                        << std::endl;
      */
      return idx;
    }
    if (j < 0 && i < (int)m_isize) {
      int idx = last_west_halo_idx() + (i + m_nhalo) + c * (m_nhalo + m_isize) +
                (-j - 1) * (m_nhalo + m_isize) * num_colors(m_ploc) +
                neigh_idx * m_nhalo * (m_nhalo + m_isize) * num_colors(m_ploc);
      /*      if (idx >= last_south_halo_idx())
              std::cout << "WARNING IN SOUTH : " << i << "," << c << "," << j <<
         ","
                        << neigh_idx << ": " << last_south_halo_idx() << " -> "
         << idx
                        << std::endl;
      */
      return idx;
    }
    if (i >= (int)m_isize && j < (int)m_jsize) {
      int idx = last_south_halo_idx() + (i - (int)m_isize) + c * m_nhalo +
                (j + (int)m_nhalo) * m_nhalo * num_colors(m_ploc) +
                neigh_idx * (m_nhalo + m_jsize) * m_nhalo * num_colors(m_ploc);
      /*      if (idx >= last_east_halo_idx())
              std::cout << "WARNING IN EAST : " << last_east_halo_idx() << " ->
         "
                        << idx << std::endl;
      */
      return idx;
    }
    if (j >= (int)m_jsize) {
      int idx =
          last_east_halo_idx() + (i + m_nhalo) + c * (m_isize + m_nhalo * 2) +
          (j - m_jsize) * num_colors(m_ploc) * (m_isize + m_nhalo * 2) +
          neigh_idx * m_nhalo * num_colors(m_ploc) * (m_isize + m_nhalo * 2);
      /*      if (idx >= last_north_halo_idx())
              std::cout << "WARNING IN NORTH : " << last_north_halo_idx() << "
         -> "
                        << idx << std::endl;
      */
      return idx;
    }
    assert(false);
    //    std::cout << "ERROR " << std::endl;
    return 0;
  }

  GT_FUNCTION
  size_t isize() const { return m_isize; }
  GT_FUNCTION
  size_t jsize() const { return m_jsize; }
  GT_FUNCTION
  size_t nhalo() const { return m_nhalo; }

private:
  size_t m_isize, m_jsize, m_nhalo;
};

class uneighbours_table : public ntable {
public:
  uneighbours_table(location primary_loc, location neigh_loc, size_t compd_size,
                    size_t totald_size)
      : ntable(primary_loc, neigh_loc,
               totald_size * num_neighbours(primary_loc, neigh_loc)),
        m_compd_size(compd_size), m_totald_size(totald_size) {}

  GT_FUNCTION
  uneighbours_table(uneighbours_table const &other)
      : m_compd_size(other.m_compd_size), m_totald_size(other.m_totald_size),
        ntable(other) {}

  GT_FUNCTION
  size_t &operator()(size_t idx, unsigned int neigh_idx) {
    return m_data[idx + m_totald_size * neigh_idx];
  }

  GT_FUNCTION
  size_t compd_size() const { return m_compd_size; }
  GT_FUNCTION
  size_t totald_size() const { return m_totald_size; }

private:
  size_t m_compd_size, m_totald_size;
};
