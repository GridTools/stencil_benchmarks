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
#include "neighbours_table.hpp"

class mesh_indexer {
public:
  mesh_indexer(location loc, size_t isize, size_t jsize, size_t nhalow,
               size_t nhalos, size_t nhaloe, size_t nhalon)
      : m_loc(loc), m_isize(isize), m_jsize(jsize), m_nhalow(nhalow),
        m_nhalos(nhalos), m_nhaloe(nhaloe), m_nhalon(nhalon) {}

private:
  location m_loc;
  size_t m_isize, m_jsize;
  size_t m_nhalow, m_nhalos, m_nhaloe, m_nhalon;

public:
  size_t last_compute_domain_idx() {
    return (m_isize)*num_colors(m_loc) * (m_jsize);
  }
  size_t last_west_halo_idx() {
    return last_compute_domain_idx() + m_jsize * m_nhalow * num_colors(m_loc);
  }
  size_t last_south_halo_idx() {
    return last_west_halo_idx() +
           m_nhalos * (m_nhalow + m_isize) * num_colors(m_loc);
  }
  size_t last_east_halo_idx() {
    return last_south_halo_idx() +
           (m_nhalos + m_jsize) * m_nhaloe * num_colors(m_loc);
  }
  size_t last_north_halo_idx() {
    return last_east_halo_idx() +
           m_nhalon * num_colors(m_loc) * (m_isize + m_nhalow * 2);
  }

  size_t index(int i, unsigned int c, int j) {
    assert(i >= -(int)m_nhalow && i < (int)(m_isize + m_nhaloe));
    assert(c < num_colors(m_loc));
    assert(j >= -(int)m_nhalos && j < (int)(m_jsize + m_nhalon));

    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize) {
      int idx = i + c * m_isize + j * num_colors(m_loc) * m_isize;
      if (idx >= last_compute_domain_idx()) {
        std::cout << "WARNING IN COMPUTE DOMAIN : " << i << "," << c << "," << j
                  << ": " << last_compute_domain_idx() << " -> " << idx
                  << std::endl;
      }
      return idx;
    }
    if (i < 0 && j >= 0 && j < (int)m_jsize) {
      int idx = (int)last_compute_domain_idx() - i - 1 + c * m_nhalow +
                j * m_nhalow * num_colors(m_loc);
      if (idx >= last_west_halo_idx())
        std::cout << "WARNING IN WEST : " << i << "," << c << "," << j << ": "
                  << last_west_halo_idx() << " -> " << idx << std::endl;

      return idx;
    }
    if (j < 0 && i < (int)m_isize) {
      int idx = last_west_halo_idx() + (i + m_nhalow) +
                c * (m_nhalow + m_isize) +
                (-j - 1) * (m_nhalow + m_isize) * num_colors(m_loc);
      if (idx >= last_south_halo_idx())
        std::cout << "WARNING IN SOUTH : " << i << "," << c << "," << j << ": "
                  << last_south_halo_idx() << " -> " << idx << std::endl;
      return idx;
    }
    if (i >= (int)m_isize && j < (int)m_jsize) {
      int idx = last_south_halo_idx() + (i - (int)m_isize) + c * m_nhaloe +
                (j + (int)m_nhalos) * m_nhaloe * num_colors(m_loc);
      if (idx >= last_east_halo_idx())
        std::cout << "WARNING IN EAST : " << last_east_halo_idx() << " -> "
                  << idx << std::endl;
      return idx;
    }
    if (j >= (int)m_jsize) {
      int idx =
          last_east_halo_idx() + (i + m_nhalow) +
          c * (m_isize + m_nhalow + m_nhaloe) +
          (j - m_jsize) * num_colors(m_loc) * (m_isize + m_nhalow + m_nhaloe);
      if (idx >= last_north_halo_idx())
        std::cout << "WARNING IN NORTH : " << last_north_halo_idx() << " -> "
                  << idx << std::endl;

      return idx;
    }
    std::cout << "ERROR " << std::endl;
    return 0;
  }
};

class elements {
  static constexpr size_t size_of_array(const location primary_loc,
                                        const unsigned int isize,
                                        const unsigned int jsize,
                                        const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo + 1) * num_colors(primary_loc) *
           sizeof(size_t);
  }

public:
  elements(location primary_loc, size_t isize, size_t jsize, size_t nhalo)
      : m_loc(primary_loc), m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        m_elements_to_cells(primary_loc, location::cell, isize, jsize, nhalo),
        m_elements_to_edges(primary_loc, location::edge, isize, jsize, nhalo),
        m_elements_to_vertices(primary_loc, location::vertex, isize, jsize,
                               nhalo),
        m_indexer(primary_loc, isize, jsize, nhalo, nhalo, nhalo, nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_idx, size_of_array(primary_loc, isize, jsize, nhalo));
#else
    m_idx = (size_t *)malloc(size_of_array(primary_loc, isize, jsize, nhalo));
#endif
  }

  size_t &neighbor(location neigh_loc, int i, unsigned int c, int j,
                   unsigned int neigh_idx) {
    if (neigh_loc == location::cell)
      return m_elements_to_cells(i, c, j, neigh_idx);
    else if (neigh_loc == location::edge)
      return m_elements_to_edges(i, c, j, neigh_idx);

    return m_elements_to_vertices(i, c, j, neigh_idx);
  }

  sneighbours_table &table(location neigh_loc) {
    if (neigh_loc == location::cell)
      return m_elements_to_cells;
    else if (neigh_loc == location::edge)
      return m_elements_to_edges;
    return m_elements_to_vertices;
  }

  size_t last_compute_domain_idx() {
    return (m_isize)*num_colors(m_loc) * (m_jsize);
  }
  size_t last_west_halo_idx() {
    return last_compute_domain_idx() + m_jsize * m_nhalo * num_colors(m_loc);
  }
  size_t last_south_halo_idx() {
    return last_west_halo_idx() +
           m_nhalo * (m_nhalo + m_isize) * num_colors(m_loc);
  }
  size_t last_east_halo_idx() {
    return last_south_halo_idx() +
           (m_nhalo + m_jsize) * m_nhalo * num_colors(m_loc);
  }
  size_t last_north_halo_idx() {
    return last_east_halo_idx() +
           m_nhalo * num_colors(m_loc) * (m_isize + m_nhalo * 2);
  }

  size_t index(int i, unsigned int c, int j) {
    assert(i >= -(int)m_nhalo && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(m_loc));
    assert(j >= -(int)m_nhalo && j < (int)(m_jsize + m_nhalo));

    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize) {
      int idx = i + c * m_isize + j * num_colors(m_loc) * m_isize;
      if (idx >= last_compute_domain_idx()) {
        std::cout << "WARNING IN COMPUTE DOMAIN : " << i << "," << c << "," << j
                  << ": " << last_compute_domain_idx() << " -> " << idx
                  << std::endl;
      }
      return idx;
    }
    if (i < 0 && j >= 0 && j < (int)m_jsize) {
      int idx = (int)last_compute_domain_idx() - i - 1 + c * m_nhalo +
                j * m_nhalo * num_colors(m_loc);
      if (idx >= last_west_halo_idx())
        std::cout << "WARNING IN WEST : " << i << "," << c << "," << j << ": "
                  << last_west_halo_idx() << " -> " << idx << std::endl;

      return idx;
    }
    if (j < 0 && i < (int)m_isize) {
      int idx = last_west_halo_idx() + (i + m_nhalo) + c * (m_nhalo + m_isize) +
                (-j - 1) * (m_nhalo + m_isize) * num_colors(m_loc);
      if (idx >= last_south_halo_idx())
        std::cout << "WARNING IN SOUTH : " << i << "," << c << "," << j << ": "
                  << last_south_halo_idx() << " -> " << idx << std::endl;
      return idx;
    }
    if (i >= (int)m_isize && j < (int)m_jsize) {
      int idx = last_south_halo_idx() + (i - (int)m_isize) + c * m_nhalo +
                (j + (int)m_nhalo) * m_nhalo * num_colors(m_loc);
      if (idx >= last_east_halo_idx())
        std::cout << "WARNING IN EAST : " << last_east_halo_idx() << " -> "
                  << idx << std::endl;
      return idx;
    }
    if (j >= (int)m_jsize) {
      int idx = last_east_halo_idx() + (i + m_nhalo) +
                c * (m_isize + m_nhalo * 2) +
                (j - m_jsize) * num_colors(m_loc) * (m_isize + m_nhalo * 2);
      if (idx >= last_north_halo_idx())
        std::cout << "WARNING IN NORTH : " << last_north_halo_idx() << " -> "
                  << idx << std::endl;

      return idx;
    }
    std::cout << "ERROR " << std::endl;
    return 0;
  }

private:
  location m_loc;
  size_t *m_idx;
  size_t m_isize, m_jsize, m_nhalo;
  sneighbours_table m_elements_to_cells, m_elements_to_edges,
      m_elements_to_vertices;
  mesh_indexer m_indexer;
};

class nodes {

  template <typename T>
  static constexpr size_t
  size_of_mesh_fields(const location loc, const unsigned int isize,
                      const unsigned int jsize, const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo + 1) * num_colors(loc) * sizeof(T);
  }

public:
  size_t totald_size() const {
    return num_nodes(m_isize, m_jsize, m_nhalo + 1);
  }

  nodes(size_t isize, size_t jsize, size_t nhalo)
      : m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        m_vertex_to_cells(location::vertex, location::cell, isize, jsize,
                          nhalo),
        m_vertex_to_edges(location::vertex, location::edge, isize, jsize,
                          nhalo),
        m_vertex_to_vertices(location::vertex, location::vertex, isize, jsize,
                             nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_x, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
    cudaMallocManaged(&m_y, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
#else
    m_x = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
    m_y = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
#endif
  }
  size_t totald_size() {
    return (m_isize + m_nhalo * 2) * (m_jsize + m_nhalo * 2);
  }

  size_t last_compute_domain_idx() {
    return (m_isize)*num_colors(location::vertex) * (m_jsize);
  }
  size_t last_ghost_east_halo_idx() {
    return last_compute_domain_idx() + m_jsize + num_colors(location::vertex);
  }
  size_t last_ghost_north_halo_idx() {
    return last_ghost_east_halo_idx() + (m_isize + 1);
  }
  size_t last_west_halo_idx() {
    return last_ghost_north_halo_idx() +
           (m_jsize + 1) * m_nhalo * num_colors(location::vertex);
  }
  size_t last_south_halo_idx() {
    return last_west_halo_idx() +
           m_nhalo * (m_nhalo + (m_isize + 1)) * num_colors(location::vertex);
  }
  size_t last_east_halo_idx() {
    return last_south_halo_idx() +
           (m_nhalo + (m_jsize + 1)) * m_nhalo * num_colors(location::vertex);
  }
  size_t last_north_halo_idx() {
    return last_east_halo_idx() +
           m_nhalo * num_colors(location::vertex) *
               ((m_isize + 1) + m_nhalo * 2);
  }

  size_t index(int i, unsigned int c, int j) {
    assert(i >= -(int)m_nhalo && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(location::vertex));
    assert(j >= -(int)m_nhalo && j < (int)(m_jsize + m_nhalo));

    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize) {
      int idx = i + j * num_colors(location::vertex) * m_isize;
      if (idx >= last_compute_domain_idx()) {
        std::cout << "WARNING IN COMPUTE DOMAIN : " << i << "," << c << "," << j
                  << ": " << last_compute_domain_idx() << " -> " << idx
                  << std::endl;
      }
      return idx;
    }
    if (i == (int)m_isize && j >= 0 && j < (int)m_jsize) {
      int idx = (int)last_compute_domain_idx() + j;
      return idx;
    }
    if (j == (int)m_jsize && i >= 0 && i < ((int)m_isize + 1)) {
      int idx = (int)last_ghost_east_halo_idx() + i;
    }
    if (i < 0 && j >= 0 && j < (int)m_jsize) {
      int idx = (int)last_ghost_north_halo_idx() - i - 1 +
                j * m_nhalo * num_colors(location::vertex);
      if (idx >= last_west_halo_idx())
        std::cout << "WARNING IN WEST : " << i << "," << c << "," << j << ": "
                  << last_west_halo_idx() << " -> " << idx << std::endl;

      return idx;
    }
    if (j < 0 && i < (int)m_isize) {
      int idx =
          last_west_halo_idx() + (i + m_nhalo) +
          (-j - 1) * (m_nhalo + (m_isize + 1)) * num_colors(location::vertex);
      if (idx >= last_south_halo_idx())
        std::cout << "WARNING IN SOUTH : " << i << "," << c << "," << j << ": "
                  << last_south_halo_idx() << " -> " << idx << std::endl;
      return idx;
    }
    if (i >= (int)m_isize && j < (int)m_jsize) {
      int idx = last_south_halo_idx() + (i - ((int)m_isize + 1)) +
                (j + (int)m_nhalo) * m_nhalo * num_colors(location::vertex);
      if (idx >= last_east_halo_idx())
        std::cout << "WARNING IN EAST : " << last_east_halo_idx() << " -> "
                  << idx << std::endl;
      return idx;
    }
    if (j >= (int)m_jsize) {
      int idx = last_east_halo_idx() + (i + m_nhalo) +
                (j - ((int)m_jsize + 1)) * num_colors(location::vertex) *
                    (((int)m_isize + 1) + 2 * m_nhalo);
      if (idx >= last_north_halo_idx())
        std::cout << "WARNING IN NORTH : " << last_north_halo_idx() << " -> "
                  << idx << std::endl;

      return idx;
    }
    std::cout << "ERROR " << std::endl;
    return 0;
  }

  size_t &neighbor(location neigh_loc, int i, unsigned int c, int j,
                   unsigned int neigh_idx) {
    if (neigh_loc == location::cell)
      return m_vertex_to_cells(i, c, j, neigh_idx);
    else if (neigh_loc == location::edge)
      return m_vertex_to_edges(i, c, j, neigh_idx);
    return m_vertex_to_vertices(i, c, j, neigh_idx);
  }

  double &x(unsigned int idx) { return m_x[idx]; }
  double &y(unsigned int idx) { return m_y[idx]; }

private:
  size_t m_isize, m_jsize, m_nhalo;
  double *m_x;
  double *m_y;
  sneighbours_table m_vertex_to_cells;
  sneighbours_table m_vertex_to_edges;
  sneighbours_table m_vertex_to_vertices;
};

//////////////// conventions ///////////////////
///  cell to vertex
/// ----------------
///
/// color 0
///
///      0
///     / \
///    /   \
///   /     \
///  2-------1
///
/// color 1
///
/// 0 -------- 1
///   \      /
///    \    /
///     \  /
///      2
///
///
///   cell to cell
///----------------
///
///  color 0
///   _________ _________
///  \        /\        /
///   \   0  /  \  1   /
///    \    /    \    /
///     \  /      \  /
///      \/________\/
///       \        /
///        \  2   /
///         \    /
///          \  /
///           \/
///
///
/// color 1
///
///           /\
///          /  \
///         /    \
///        /  2   \
///       /________\
///      /\        /\
///     /  \      /  \
///    /    \    /    \
///   /   1  \  /  0   \
///  /________\/________\
///
///

class mesh {

  size_t end_idx_compute_domain() { return m_isize * m_jsize; }

public:
  nodes &get_nodes() { return m_nodes; }

  elements &get_elements(location loc) {
    return (loc == location::cell) ? m_cells : m_edges;
  }

  size_t compd_size() { return m_isize * m_jsize; }

  size_t totald_size() {
    return (m_isize + m_nhalo * 2) * (m_jsize + m_nhalo * 2);
  }

  size_t nodes_totald_size() const { return m_nodes.totald_size(); }
  size_t isize() const { return m_isize; }
  size_t jsize() const { return m_jsize; }
  size_t nhalo() const { return m_nhalo; }

  mesh(const size_t isize, const size_t jsize, const size_t nhalo)
      : m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        m_i_domain{0, (int)isize}, m_j_domain{0, (int)jsize},
        m_cells(location::cell, isize, jsize, nhalo),
        m_edges(location::edge, isize, jsize, nhalo),
        m_nodes(isize, jsize, nhalo) {

    for (size_t i = 0; i < isize; ++i) {
      for (size_t j = 0; j < jsize; ++j) {
        // cell to cell
        m_cells.neighbor(location::cell, i, 0, j, 0) =
            m_cells.index(i - 1, 1, j);
        m_cells.neighbor(location::cell, i, 0, j, 1) = m_cells.index(i, 1, j);
        m_cells.neighbor(location::cell, i, 0, j, 2) =
            m_cells.index(i, 1, j - 1);

        m_cells.neighbor(location::cell, i, 1, j, 0) =
            m_cells.index(i + 1, 0, j);
        m_cells.neighbor(location::cell, i, 1, j, 1) = m_cells.index(i, 0, j);
        m_cells.neighbor(location::cell, i, 1, j, 2) =
            m_cells.index(i, 0, j + 1);

        // cell to edge
        m_cells.neighbor(location::edge, i, 0, j, 0) = m_edges.index(i, 1, j);
        m_cells.neighbor(location::edge, i, 0, j, 1) = m_edges.index(i, 2, j);
        m_cells.neighbor(location::edge, i, 0, j, 2) = m_edges.index(i, 0, j);

        m_cells.neighbor(location::edge, i, 1, j, 0) =
            m_edges.index(i + 1, 1, j);
        m_cells.neighbor(location::edge, i, 1, j, 1) = m_edges.index(i, 2, j);
        m_cells.neighbor(location::edge, i, 1, j, 2) =
            m_edges.index(i, 0, j + 1);

        m_cells.neighbor(location::vertex, i, 0, j, 0) =
            m_nodes.index(i, 0, j + 1);

        m_cells.neighbor(location::vertex, i, 0, j, 1) =
            m_nodes.index(i + 1, 0, j);
        m_cells.neighbor(location::vertex, i, 0, j, 2) = m_nodes.index(i, 0, j);

        m_cells.neighbor(location::vertex, i, 1, j, 0) =
            m_nodes.index(i, 0, j + 1);

        m_cells.neighbor(location::vertex, i, 1, j, 1) =
            m_nodes.index(i + 1, 0, j + 1);

        m_cells.neighbor(location::vertex, i, 1, j, 2) =
            m_nodes.index(i + 1, 0, j);

        // edge to edge
        m_edges.neighbor(location::edge, i, 0, j, 0) =
            m_edges.index(i, 2, j - 1);
        m_edges.neighbor(location::edge, i, 0, j, 1) = m_edges.index(i, 1, j);
        m_edges.neighbor(location::edge, i, 0, j, 2) =
            m_edges.index(i + 1, 1, j - 1);
        m_edges.neighbor(location::edge, i, 0, j, 3) = m_edges.index(i, 2, j);
        m_edges.neighbor(location::edge, i, 1, j, 0) = m_edges.index(i, 0, j);
        m_edges.neighbor(location::edge, i, 1, j, 1) =
            m_edges.index(i - 1, 2, j);
        m_edges.neighbor(location::edge, i, 1, j, 2) =
            m_edges.index(i - 1, 0, j + 1);
        m_edges.neighbor(location::edge, i, 1, j, 3) = m_edges.index(i, 2, j);
        m_edges.neighbor(location::edge, i, 2, j, 0) = m_edges.index(i, 0, j);
        m_edges.neighbor(location::edge, i, 2, j, 1) = m_edges.index(i, 1, j);
        m_edges.neighbor(location::edge, i, 2, j, 2) =
            m_edges.index(i + 1, 1, j);
        m_edges.neighbor(location::edge, i, 2, j, 3) =
            m_edges.index(i, 0, j + 1);

        m_edges.neighbor(location::cell, i, 0, j, 0) =
            m_cells.index(i, 1, j - 1);
        m_edges.neighbor(location::cell, i, 0, j, 1) = m_cells.index(i, 0, j);
        m_edges.neighbor(location::cell, i, 1, j, 0) =
            m_cells.index(i - 1, 1, j);
        m_edges.neighbor(location::cell, i, 1, j, 1) = m_cells.index(i, 0, j);

        m_edges.neighbor(location::cell, i, 2, j, 0) = m_cells.index(i, 1, j);
        m_edges.neighbor(location::cell, i, 2, j, 1) = m_cells.index(i, 0, j);

        m_edges.neighbor(location::vertex, i, 0, j, 0) =
            m_nodes.index(i + 1, 0, j);

        m_edges.neighbor(location::vertex, i, 0, j, 1) = m_nodes.index(i, 0, j);

        m_edges.neighbor(location::vertex, i, 1, j, 0) = m_nodes.index(i, 0, j);

        m_edges.neighbor(location::vertex, i, 1, j, 1) =
            m_nodes.index(i, 0, j + 1);

        m_edges.neighbor(location::vertex, i, 2, j, 0) =
            m_nodes.index(i, 0, j + 1);

        m_edges.neighbor(location::vertex, i, 2, j, 1) =
            m_nodes.index(i + 1, 0, j);

        m_nodes.neighbor(location::vertex, i, 0, j, 0) =
            m_nodes.index(i, 0, j - 1);
        m_nodes.neighbor(location::vertex, i, 0, j, 1) =
            m_nodes.index(i, 0, j + 1);
        m_nodes.neighbor(location::vertex, i, 0, j, 2) =
            m_nodes.index(i + 1, 0, j);
        m_nodes.neighbor(location::vertex, i, 0, j, 3) =
            m_nodes.index(i - 1, 0, j);
        m_nodes.neighbor(location::vertex, i, 0, j, 4) =
            m_nodes.index(i + 1, 0, j - 1);
        m_nodes.neighbor(location::vertex, i, 0, j, 5) =
            m_nodes.index(i - 1, 0, j + 1);

        m_nodes.neighbor(location::edge, i, 0, j, 0) =
            m_edges.index(i, 1, j - 1);
        m_nodes.neighbor(location::edge, i, 0, j, 1) =
            m_edges.index(i - 1, 0, j);
        m_nodes.neighbor(location::edge, i, 0, j, 2) =
            m_edges.index(i - 1, 2, j);
        m_nodes.neighbor(location::edge, i, 0, j, 3) = m_edges.index(i, 1, j);
        m_nodes.neighbor(location::edge, i, 0, j, 4) = m_edges.index(i, 0, j);
        m_nodes.neighbor(location::edge, i, 0, j, 5) =
            m_edges.index(i, 2, j - 1);

        m_nodes.x(index(location::vertex, i, 0, j)) = i + j * 0.5;
        m_nodes.y(index(location::vertex, i, 0, j)) = j;
      }
    }
    // add first line artificial nodes halo on the East
    m_curr_idx = end_idx_compute_domain();
    for (size_t j = 0; j < jsize; ++j) {
      m_cells.neighbor(location::vertex, isize - 1, 0, j, 1) = m_curr_idx + j;

      m_cells.neighbor(location::vertex, isize - 1, 1, j, 2) = m_curr_idx + j;

      if (j > 0)
        m_cells.neighbor(location::vertex, isize - 1, 1, j - 1, 1) =
            m_curr_idx + j;

      m_cells.neighbor(location::vertex, isize - 1, 0, j, 1) = m_curr_idx + j;

      m_cells.neighbor(location::vertex, isize - 1, 1, j, 2) = m_curr_idx + j;

      if (j > 0)
        m_cells.neighbor(location::vertex, isize - 1, 1, j - 1, 1) =
            m_curr_idx + j;

      m_edges.neighbor(location::vertex, isize - 1, 0, j, 0) = m_curr_idx + j;
      m_edges.neighbor(location::vertex, isize - 1, 2, j, 1) = m_curr_idx + j;

      m_nodes.x(m_curr_idx + j) = m_isize + j * 0.5;
      m_nodes.y(m_curr_idx + j) = j;
    }
    m_curr_idx += jsize;
    //    m_i_domain[1]++;

    // add first line artificial nodes halo on the North
    for (int i = m_i_domain[0]; i < m_i_domain[1]; ++i) {
      m_cells.neighbor(location::vertex, i, 0, jsize - 1, 0) = m_curr_idx + i;

      m_cells.neighbor(location::vertex, i, 1, jsize - 1, 0) = m_curr_idx + i;
      if (i >= 0) {
        m_cells.neighbor(location::vertex, i, 1, jsize - 1, 1) =
            m_curr_idx + i + 1;
      }
      if (i == m_i_domain[i] - 1)
        std::cout << m_curr_idx + i << std::endl;

      m_edges.neighbor(location::vertex, i, 1, jsize - 1, 1) = m_curr_idx + i;
      m_edges.neighbor(location::vertex, i, 2, jsize - 1, 0) = m_curr_idx + i;
      m_nodes.x(m_curr_idx + i) = i + m_jsize * 0.5;
      m_nodes.y(m_curr_idx + i) = m_jsize;
    }

    m_curr_idx += m_i_domain[1] - m_i_domain[0];
    m_nodes.x(m_curr_idx) = m_i_domain[1] - m_i_domain[0] + m_jsize * 0.5;
    m_nodes.y(m_curr_idx) = m_jsize;
    m_curr_idx++;

    //    m_j_domain[1]++;

    // add lines real halo on the West
    for (int c = 0; c < m_nhalo; ++c) {

      for (int j = m_j_domain[0]; j < m_j_domain[1]; ++j) {
        m_cells.neighbor(location::vertex, -1 - c, 1, j, 0) =
            m_curr_idx + j + 1;
        m_cells.neighbor(location::vertex, -1 - c, 1, j, 1) =
            m_cells.neighbor(location::vertex, -c, 1, j, 0);
        m_cells.neighbor(location::vertex, -1 - c, 1, j, 2) =
            m_cells.neighbor(location::vertex, -c, 0, j, 2);

        m_cells.neighbor(location::vertex, -1 - c, 0, j, 0) =
            m_curr_idx + j + 1;
        m_cells.neighbor(location::vertex, -1 - c, 0, j, 1) =
            m_cells.neighbor(location::vertex, -c, 0, j, 2);
        m_cells.neighbor(location::vertex, -1 - c, 0, j, 2) = m_curr_idx + j;

        m_nodes.x(m_curr_idx + (j - m_j_domain[0])) = j * 0.5 - 1 - c;
        m_nodes.y(m_curr_idx + (j - m_j_domain[0])) = j;
      }
      m_curr_idx += m_j_domain[1] - m_j_domain[0];

      m_nodes.x(m_curr_idx) = m_j_domain[1] * 0.5 - 1 - c;
      m_nodes.y(m_curr_idx) = m_j_domain[1];
      m_curr_idx++;

      m_i_domain[0]--;
    }

    // add first line real halo on the South
    for (int c = 0; c < m_nhalo; ++c) {

      for (int i = m_i_domain[0]; i < m_i_domain[1]; ++i) {
        m_cells.neighbor(location::vertex, i, 1, -1 - c, 0) =
            m_cells.neighbor(location::vertex, i, 0, -c, 2);
        m_cells.neighbor(location::vertex, i, 1, -1 - c, 1) =
            m_cells.neighbor(location::vertex, i, 0, -c, 1);
        m_cells.neighbor(location::vertex, i, 1, -1 - c, 2) =
            m_curr_idx + (i - m_i_domain[0]) + 1;
        if (i == -1 && c == 1)
          std::cout << "IND " << c << " " << m_cells.index(i, 0, -1 - c) << " "
                    << m_cells.neighbor(location::vertex, i, 0, -c, 2) << " "
                    << m_cells.neighbor(location::vertex, i, 0, -c, 1) << " "
                    << m_curr_idx + (i - m_i_domain[0]) + 1 << std::endl;
        m_cells.neighbor(location::vertex, i, 0, -1 - c, 0) =
            m_cells.neighbor(location::vertex, i, 0, -c, 2);
        m_cells.neighbor(location::vertex, i, 0, -1 - c, 1) =
            m_curr_idx + (i - m_i_domain[0]) + 1;
        m_cells.neighbor(location::vertex, i, 0, -1 - c, 2) =
            m_curr_idx + (i - m_i_domain[0]);

        m_nodes.x(m_curr_idx + (i - m_i_domain[0])) = i - 0.5 * (1 + c);
        m_nodes.y(m_curr_idx + (i - m_i_domain[0])) = -1 - c;
      }
      m_curr_idx += m_i_domain[1] - m_i_domain[0];

      m_nodes.x(m_curr_idx) = m_i_domain[1] - 0.5 * (1 + c);
      m_nodes.y(m_curr_idx) = -1 - c;

      m_curr_idx++;

      m_j_domain[0]--;
    }
    // add first line real halo on the East
    for (int c = 0; c < m_nhalo; ++c) {

      for (int j = m_j_domain[0]; j < m_j_domain[1]; ++j) {
        int i = m_isize + c;
        m_cells.neighbor(location::vertex, i, 1, j, 0) =
            m_cells.neighbor(location::vertex, i - 1, 1, j, 1);
        m_cells.neighbor(location::vertex, i, 1, j, 1) =
            m_curr_idx + (j - m_j_domain[0]) + 1;
        m_cells.neighbor(location::vertex, i, 1, j, 2) =
            m_curr_idx + (j - m_j_domain[0]);

        m_cells.neighbor(location::vertex, i, 0, j, 0) =
            m_cells.neighbor(location::vertex, i - 1, 1, j, 1);
        m_cells.neighbor(location::vertex, i, 0, j, 1) =
            m_curr_idx + (j - m_j_domain[0]);
        m_cells.neighbor(location::vertex, i, 0, j, 2) =
            m_cells.neighbor(location::vertex, i - 1, 0, j, 1);

        m_nodes.x(m_curr_idx + (j - m_j_domain[0])) =
            i + (j - m_j_domain[0]) * 0.5;
        m_nodes.y(m_curr_idx + (j - m_j_domain[0])) = (j);
      }
      m_curr_idx += m_j_domain[1] - m_j_domain[0];

      m_nodes.x(m_curr_idx) = m_i_domain[1] + m_j_domain[1] * 0.5 + 1;
      m_nodes.y(m_curr_idx) = m_j_domain[1];

      m_curr_idx++;

      m_i_domain[1]++;
    }

    // add first line real halo on the North
    for (int c = 0; c < m_nhalo; ++c) {

      for (int i = m_i_domain[0]; i < m_i_domain[1]; ++i) {
        int j = m_jsize + c;
        m_cells.neighbor(location::vertex, i, 1, j, 0) =
            m_curr_idx + (i - m_i_domain[0]);
        m_cells.neighbor(location::vertex, i, 1, j, 1) =
            m_curr_idx + (i - m_i_domain[0]) + 1;
        m_cells.neighbor(location::vertex, i, 1, j, 2) =
            m_cells.neighbor(location::vertex, i, 1, j - 1, 1);

        m_cells.neighbor(location::vertex, i, 0, j, 0) =
            m_curr_idx + (i - m_i_domain[0]);
        m_cells.neighbor(location::vertex, i, 0, j, 1) =
            m_cells.neighbor(location::vertex, i, 1, j - 1, 1);
        m_cells.neighbor(location::vertex, i, 0, j, 2) =
            m_cells.neighbor(location::vertex, i, 1, j - 1, 0);

        m_nodes.x(m_curr_idx + (i - m_i_domain[0])) =
            i + 0.5 + m_j_domain[1] * 0.5;
        m_nodes.y(m_curr_idx + (i - m_i_domain[0])) = m_j_domain[1] + 1;
      }
      m_curr_idx += m_i_domain[1] - m_i_domain[0];
      m_nodes.x(m_curr_idx) = m_i_domain[1] + (m_j_domain[1] + 1) * 0.5;
      m_nodes.y(m_curr_idx) = m_j_domain[1] + 1;

      m_curr_idx++;

      m_j_domain[1]++;
    }
  }
  size_t element_index(location primary_loc, int i, unsigned int c, int j) {
    assert(i > -(int)(m_nhalo) && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(primary_loc));
    assert(j > -(int)(m_nhalo) && j < (int)(m_jsize + m_nhalo));

    return (i + (int)m_nhalo) + c * (m_isize + m_nhalo * 2) +
           (j + (int)m_nhalo) * (m_isize + m_nhalo * 2) *
               num_colors(primary_loc);
  }

  size_t index(location primary_loc, unsigned int i, unsigned int c,
               unsigned int j) {
    assert(i < m_isize);
    assert(c < num_colors(primary_loc));
    assert(j < m_jsize);

    return i + c * (m_isize) + j * (m_isize)*num_colors(primary_loc);
  }

  void test() {
    int idx = 0;
    for (int n = 0; n < num_neighbours(location::cell, location::vertex); ++n) {
      idx = 0;
      for (int j = 0; j < (int)m_jsize; ++j) {
        for (int c = 0; c < num_colors(location::cell); ++c) {
          for (int i = 0; i < (int)m_isize; ++i) {
            if (m_cells.table(location::vertex).index_in_tables(i, c, j, n) !=
                m_cells.table(location::vertex).neigh_index(idx, n))
              std::cout << "ERROR " << i << " " << c << " " << j << " " << n
                        << " ; "
                        << m_cells.table(location::vertex)
                               .index_in_tables(i, c, j, n)
                        << " -> [" << idx << "]"
                        << m_cells.table(location::vertex).neigh_index(idx, n)
                        << std::endl;
            ++idx;
          }
        }
      }
    }

    for (int j = 1; j < (int)m_jsize - 1; ++j) {
      for (int i = 1; i < (int)m_isize - 1; ++i) {
        if ((m_cells.table(location::cell)(i, 0, j, 0) !=
             m_cells.index(i - 1, 1, j)) ||
            (m_cells.table(location::cell)(i, 0, j, 1) !=
             m_cells.index(i, 1, j)) ||
            (m_cells.table(location::cell)(i, 0, j, 2) !=
             m_cells.index(i, 1, j - 1)))
          std::cout << "ERROR " << i << " "
                    << " " << j << " "
                    << " ; " << m_cells.table(location::cell)(i, 0, j, 0) << " "
                    << m_cells.index(i, 1, j) << " " << m_cells.index(i, 0, j)
                    << std::endl;
        ++idx;
      }
    }
  }

  void print() {
    std::stringstream ss;
    ss << "$MeshFormat" << std::endl
       << "2.2 0 8" << std::endl
       << "$EndMeshFormat" << std::endl;

    ss << "$Nodes" << std::endl;
    ss << m_curr_idx << std::endl; // num_nodes(m_isize + 1, m_jsize + 1, 0)
                                   // + 1 << std::endl;

    for (size_t i = 0; i < m_curr_idx; ++i) {
      ss << i + 1 << " " << m_nodes.x(i) << " " << m_nodes.y(i) << " 1 "
         << std::endl;
    }

    ss << "$EndNodes" << std::endl;
    ss << "$Elements" << std::endl;
    //    ss << num_nodes(m_isize + 3, m_jsize + 3, 0) * 2 // edges
    ss << num_nodes(m_isize + m_nhalo * 2, m_jsize + m_nhalo * 2, 0) *
              2 // edges
       // + num_nodes(m_isize, m_jsize, 0) * 3
       << std::endl;

    for (int j = -m_nhalo; j < (int)m_jsize + (int)m_nhalo; ++j) {
      for (int i = -m_nhalo; i < (int)m_isize + (int)m_nhalo; ++i) {

        ss << m_cells.index(i, 0, j) + 1 << " 2 4 1 1 1 28 "
           << m_cells.neighbor(location::vertex, i, 0, j, 0) + 1 << " "
           << m_cells.neighbor(location::vertex, i, 0, j, 1) + 1 << " "
           << m_cells.neighbor(location::vertex, i, 0, j, 2) + 1 << std::endl;

        ss << m_cells.index(i, 1, j) + 1 << " 2 4 1 1 1 28 "
           << m_cells.neighbor(location::vertex, i, 1, j, 0) + 1 << " "
           << m_cells.neighbor(location::vertex, i, 1, j, 1) + 1 << " "
           << m_cells.neighbor(location::vertex, i, 1, j, 2) + 1 << std::endl;
      }
    }

    //    // print the edges
    //    for (size_t j = 0; j < m_jsize; ++j) {
    //      for (size_t i = 0; i < m_isize; ++i) {
    //        ss << index(location::edge, i, 0, j) + 1 + 10000 << " 1 4 1 1
    //        1 28
    //        "
    //           << m_edge_to_vertices(i, 0, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 0, j, 1) + 1 << std::endl;

    //        ss << index(location::edge, i, 1, j) + 1 + 10000 << " 1 4 1 1
    //        1 28
    //        "
    //           << m_edge_to_vertices(i, 1, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 1, j, 1) + 1 << std::endl;
    //        ss << index(location::edge, i, 2, j) + 1 + 10000 << " 1 4 1  1
    //        1
    //        28 "
    //           << m_edge_to_vertices(i, 2, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 2, j, 1) + 1 << std::endl;
    //      }
    //    }

    ss << "$EndElements" << std::endl;
    std::ofstream msh_file;
    msh_file.open("mesh.gmsh");
    msh_file << ss.str();
    msh_file.close();
  }

private:
  size_t m_isize, m_jsize, m_nhalo;
  std::array<int, 2> m_i_domain, m_j_domain;
  size_t m_curr_idx;

  nodes m_nodes;
  elements m_cells;
  elements m_edges;
};
