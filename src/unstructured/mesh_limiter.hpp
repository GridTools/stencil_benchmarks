#pragma once
#include <math.h>
#include <unordered_map>
#include "mesh.hpp"
#include "umesh.hpp"
#include "hilbert.hpp"

size_t get_cell_idx(std::unordered_map<size_t, size_t> &halo_idxs_pairs,
                    size_t cell_idx, size_t &halo_idx) {

  if (!halo_idxs_pairs.count(cell_idx)) {
    halo_idxs_pairs[cell_idx] = halo_idx++;
  }
  return halo_idxs_pairs[cell_idx];
}

void fill_halo_cells(std::unordered_map<size_t, size_t> &halo_idxs_pairs,
                     mesh &mesh_, uelements &ucells, int i, int c, int j,
                     size_t isize, size_t jsize, size_t nhalo, size_t &halo_idx,
                     bool ende = false) {
  size_t cell_idx =
      get_cell_idx(halo_idxs_pairs,
                   mesh_.get_elements(location::cell).index(i, c, j), halo_idx);
  for (size_t n = 0; n < num_neighbours(location::cell, location::vertex);
       ++n) {

    ucells.table(location::vertex)(cell_idx, n) =
        mesh_.get_elements(location::cell)
            .neighbor(location::vertex, i, c, j, n);
  }

  for (size_t n = 0; n < num_neighbours(location::cell, location::cell); ++n) {
    if (j == (int)jsize + (int)nhalo - 1 && c == 1 && n == 2)
      continue;
    if (i == (int)isize + (int)nhalo - 1 && c == 1 && n == 0)
      continue;
    if (i == -(int)nhalo && c == 0 && n == 2)
      continue;
    if (j == -(int)nhalo && c == 0 && n == 0)
      continue;

    if (mesh_.get_elements(location::cell)
            .neighbor(location::cell, i, c, j, n) == 0)
      continue;
    size_t nv_idx =
        get_cell_idx(halo_idxs_pairs, mesh_.get_elements(location::cell)
                                          .neighbor(location::cell, i, c, j, n),
                     halo_idx);

    ucells.table(location::cell)(cell_idx, n) = nv_idx;
  }
}

void limit_mesh(umesh &umesh_, mesh &mesh_, double xmin, double ymin,
                double xmax, double ymax) {

  assert(mesh_.isize() == mesh_.jsize());

  const double number_div = log2((float)mesh_.isize());
  double intpart;
  assert(std::modf(number_div, &intpart) == 0.0);

  std::vector<std::array<int, 2>> inds;
  hilbert(inds, 0, 0, mesh_.isize(), 0, 0, mesh_.jsize(), number_div);

  size_t halo_idx = mesh_.compd_size() * num_colors(location::cell);
  std::unordered_map<size_t, size_t> halo_idxs_pairs;

  size_t isize = mesh_.isize();
  size_t jsize = mesh_.jsize();

  auto ucells = umesh_.get_elements(location::cell);

  for (size_t idx = 0; idx != inds.size(); ++idx) {
    int i = inds[idx][0];
    int j = inds[idx][1];

    if (idx % 1000 == 0)
      // color 0
      if (i > 0) {
        auto pos =
            std::find(inds.begin(), inds.end(), std::array<int, 2>{i - 1, j});
        assert(pos != std::end(inds));

        ucells.table(location::cell)(idx * 2, 0) =
            std::distance(inds.begin(), pos) * 2 + 1;

      } else {
        ucells.table(location::cell)(idx * 2, 0) = halo_idx;
        halo_idxs_pairs[mesh_.get_elements(location::cell)
                            .neighbor(location::cell, i, 0, j, 0)] = halo_idx;
        halo_idx++;
      }

    {
      auto pos = std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j});
      assert(pos != std::end(inds));

      ucells.table(location::cell)(idx * 2, 1) =
          std::distance(inds.begin(), pos) * 2 + 1;
    }
    if (j > 0) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j - 1});
      assert(pos != std::end(inds));

      ucells.table(location::cell)(idx * 2, 2) =
          std::distance(inds.begin(), pos) * 2 + 1;
    } else {
      ucells.table(location::cell)(idx * 2, 2) = halo_idx;

      halo_idxs_pairs[mesh_.get_elements(location::cell)
                          .neighbor(location::cell, i, 0, j, 2)] = halo_idx;
      halo_idx++;
    }

    // color 1
    if (i < isize - 1) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i + 1, j});
      assert(pos != std::end(inds));

      ucells.table(location::cell)(idx * 2 + 1, 0) =
          std::distance(inds.begin(), pos) * 2;
    } else {

      ucells.table(location::cell)(idx * 2 + 1, 0) = halo_idx;
      halo_idxs_pairs[mesh_.get_elements(location::cell)
                          .neighbor(location::cell, i, 1, j, 0)] = halo_idx;
      halo_idx++;
    }

    {
      auto pos = std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j});
      assert(pos != std::end(inds));

      ucells.table(location::cell)(idx * 2 + 1, 1) =
          std::distance(inds.begin(), pos) * 2;
    }
    if (j < jsize - 1) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j + 1});
      assert(pos != std::end(inds));

      ucells.table(location::cell)(idx * 2 + 1, 2) =
          std::distance(inds.begin(), pos) * 2;
    } else {
      ucells.table(location::cell)(idx * 2 + 1, 2) = halo_idx;
      halo_idxs_pairs[mesh_.get_elements(location::cell)
                          .neighbor(location::cell, i, 1, j, 2)] = halo_idx;
      halo_idx++;
    }

    for (size_t n = 0; n < num_neighbours(location::cell, location::vertex);
         ++n) {
      ucells.table(location::vertex)(idx * 2, n) =
          mesh_.get_elements(location::cell)
              .neighbor(location::vertex, i, 0, j, n);
      ucells.table(location::vertex)(idx * 2 + 1, n) =
          mesh_.get_elements(location::cell)
              .neighbor(location::vertex, i, 1, j, n);
    }
  }

  for (int i = -mesh_.nhalo(); i < 0; ++i) {
    for (size_t c = 0; c < num_colors(location::cell); ++c) {
      for (size_t j = 0; j < mesh_.jsize(); ++j) {
        fill_halo_cells(halo_idxs_pairs, mesh_, ucells, i, c, j, mesh_.isize(),
                        mesh_.jsize(), mesh_.nhalo(), halo_idx);
      }
    }
  }

  for (size_t i = mesh_.isize(); i < mesh_.isize() + mesh_.nhalo(); ++i) {
    for (size_t c = 0; c < num_colors(location::cell); ++c) {
      for (size_t j = 0; j < mesh_.jsize(); ++j) {
        fill_halo_cells(halo_idxs_pairs, mesh_, ucells, i, c, j, mesh_.isize(),
                        mesh_.jsize(), mesh_.nhalo(), halo_idx);
      }
    }
  }

  for (int i = -(int)mesh_.nhalo(); i < (int)mesh_.isize() + (int)mesh_.nhalo();
       ++i) {
    for (size_t c = 0; c < num_colors(location::cell); ++c) {
      for (int j = -(int)mesh_.nhalo(); j < 0; ++j) {
        fill_halo_cells(halo_idxs_pairs, mesh_, ucells, i, c, j, mesh_.isize(),
                        mesh_.jsize(), mesh_.nhalo(), halo_idx);
      }
    }
  }

  for (int i = -(int)mesh_.nhalo(); i < (int)mesh_.isize() + (int)mesh_.nhalo();
       ++i) {
    for (size_t c = 0; c < num_colors(location::cell); ++c) {
      for (int j = (int)mesh_.jsize();
           j < (int)mesh_.jsize() + (int)mesh_.nhalo(); ++j) {
        fill_halo_cells(halo_idxs_pairs, mesh_, ucells, i, c, j, mesh_.isize(),
                        mesh_.jsize(), mesh_.nhalo(), halo_idx, true);
      }
    }
  }

  for (size_t cnt = 0; cnt != mesh_.nodes_totald_size(); ++cnt) {
    umesh_.get_nodes().x(cnt) = mesh_.get_nodes().x(cnt);
    umesh_.get_nodes().y(cnt) = mesh_.get_nodes().y(cnt);
  }

  //  for (size_t i = 0; i < m_isize; ++i) {
  //    for (size_t j = 0; j < m_jsize; ++j) {
  //    }
  //  }
}
