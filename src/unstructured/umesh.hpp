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

class uelements {
public:
  uelements(location primary_loc, size_t compd_size, size_t totald_size)
      : m_loc(primary_loc), m_compd_size(compd_size),
        m_totald_size(totald_size),
        m_elements_to_cells(primary_loc, location::cell, compd_size,
                            totald_size),
        m_elements_to_edges(primary_loc, location::edge, compd_size,
                            totald_size),
        m_elements_to_vertices(primary_loc, location::vertex, compd_size,
                               totald_size) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_idx, totald_size * sizeof(size_t));
#else
    m_idx = (size_t *)malloc(totald_size * sizeof(size_t));
#endif
  }

  uneighbours_table &table(location neigh_loc) {
    if (neigh_loc == location::cell)
      return m_elements_to_cells;
    else if (neigh_loc == location::edge)
      return m_elements_to_edges;
    return m_elements_to_vertices;
  }

  GT_FUNCTION
  size_t totald_size() { return m_totald_size; }

private:
  location m_loc;
  size_t *m_idx;
  size_t m_compd_size, m_totald_size;
  uneighbours_table m_elements_to_cells, m_elements_to_edges,
      m_elements_to_vertices;
};

class unodes {

public:
  unodes(size_t compd_size, size_t totald_size)
      : m_compd_size(compd_size), m_totald_size(totald_size),
        m_vertex_to_cells(location::vertex, location::cell, compd_size,
                          totald_size),
        m_vertex_to_edges(location::vertex, location::edge, compd_size,
                          totald_size),
        m_vertex_to_vertices(location::vertex, location::vertex, compd_size,
                             totald_size) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_x, totald_size * sizeof(double));
    cudaMallocManaged(&m_y, totald_size * sizeof(double));
#else
    m_x = (double *)malloc(totald_size * sizeof(double));
    m_y = (double *)malloc(totald_size * sizeof(double));
#endif
  }

  double &x(unsigned int idx) { return m_x[idx]; }
  double &y(unsigned int idx) { return m_y[idx]; }
  GT_FUNCTION
  size_t totald_size() const { return m_totald_size; }
  size_t compd_size() const { return m_compd_size; }

private:
  size_t m_compd_size, m_totald_size;
  double *m_x;
  double *m_y;
  uneighbours_table m_vertex_to_cells;
  uneighbours_table m_vertex_to_edges;
  uneighbours_table m_vertex_to_vertices;
};

////////////////// conventions ///////////////////
///// cell to vertex
///// 0 -------- 1
/////   \      /
/////    \    /
/////     \  /
/////      2

/////
/////      0
/////     / \
/////    /   \
/////   /     \
/////  2-------1

class umesh {

public:
  unodes &get_nodes() { return m_nodes; }

  uelements &get_elements(location loc) {
    return (loc == location::cell) ? m_cells : m_edges;
  }

  umesh() = delete;

  umesh(const size_t compd_size, const size_t totald_size,
        const size_t nodes_totald_size)
      : m_compd_size(compd_size), m_totald_size(totald_size),
        m_nodes_totald_size(nodes_totald_size),
        m_cells(location::cell, compd_size * num_colors(location::cell),
                totald_size * num_colors(location::cell)),
        m_edges(location::edge, compd_size * num_colors(location::edge),
                totald_size * num_colors(location::edge)),
        m_nodes(compd_size * num_colors(location::vertex),
                nodes_totald_size * num_colors(location::vertex)) {}

  GT_FUNCTION 
  size_t compd_size() { return m_compd_size; }
  GT_FUNCTION
  size_t totald_size() { return m_totald_size; }

  void print() {
    std::stringstream ss;
    ss << "$MeshFormat" << std::endl
       << "2.2 0 8" << std::endl
       << "$EndMeshFormat" << std::endl;

    ss << "$Nodes" << std::endl;
    ss << m_nodes.totald_size() << std::endl;

    for (size_t i = 0; i < m_nodes.totald_size(); ++i) {
      ss << i + 1 << " " << m_nodes.x(i) << " " << m_nodes.y(i) << " 1 "
         << std::endl;
    }

    ss << "$EndNodes" << std::endl;
    ss << "$Elements" << std::endl;
    //    ss << num_nodes(m_isize + 3, m_jsize + 3, 0) * 2 // edges
    auto cell_to_vertex = m_cells.table(location::vertex);

    ss << cell_to_vertex.totald_size() << std::endl;

    for (size_t idx = 0; idx < cell_to_vertex.totald_size(); ++idx) {

      ss << idx + 1 << " 2 4 1 1 1 28 " << cell_to_vertex(idx, 0) + 1 << " "
         << cell_to_vertex(idx, 1) + 1 << " " << cell_to_vertex(idx, 2) + 1
         << std::endl;
    }

    ss << "$EndElements" << std::endl;
    std::ofstream msh_file;
    msh_file.open("umesh.gmsh");
    msh_file << ss.str();
    msh_file.close();
  }

private:
  size_t m_compd_size, m_totald_size, m_nodes_totald_size;

  unodes m_nodes;
  uelements m_cells;
  uelements m_edges;
};
