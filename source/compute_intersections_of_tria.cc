// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Luca Heltai
//
// This file is part of the FSI-suite platform, based on the deal.II library.
//
// The FSI-suite platform is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3.0 of the License,
// or (at your option) any later version. The full text of the license can be
// found in the file LICENSE at the top level of the FSI-suite platform
// distribution.
//
// ---------------------------------------------------------------------

#include "compute_intersections_of_tria.h"


using namespace dealii;
namespace dealii::NonMatching
{
  template <int dim0, int dim1, int spacedim>
  std::vector<
    std::tuple<typename Triangulation<dim0, spacedim>::active_cell_iterator,
               typename Triangulation<dim1, spacedim>::active_cell_iterator,
               Quadrature<spacedim>>>
  intersect_and_get_quads(
    const GridTools::Cache<dim0, spacedim> &immersed_cache,
    const GridTools::Cache<dim1, spacedim> &space_cache,
    const unsigned int                      degree)
  {
    Assert(degree >= 1, ExcMessage("degree cannot be 0"));
    std::set<typename Triangulation<dim1, spacedim>::active_cell_iterator>
                intersected_cells; // avoid duplicates
    const auto &space_tree =
      space_cache.get_locally_owned_cell_bounding_boxes_rtree();
    const auto &immersed_tree =
      immersed_cache.get_locally_owned_cell_bounding_boxes_rtree();


    namespace bgi = boost::geometry::index;
    // Whenever the BB space_cell intersects the BB of an embedded cell, store
    // the space_cell in the set of intersected_cells
    for (const auto &[immersed_box, immersed_cell] : immersed_tree)
      {
        for (const auto &[space_box, space_cell] :
             space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box)))
          {
            intersected_cells.insert(space_cell);
          }
      } // found intersected cells


    // references to triangulations' info (cp cstrs marked as delete)
    const auto &immersed_grid = immersed_cache.get_triangulation();
    const auto &mapping0      = immersed_cache.get_mapping();
    const auto &mapping1      = space_cache.get_mapping();

    std::vector<
      std::tuple<typename Triangulation<dim0, spacedim>::active_cell_iterator,
                 typename Triangulation<dim1, spacedim>::active_cell_iterator,
                 Quadrature<spacedim>>>
      cells_with_quads;

    for (const auto &space_cell : intersected_cells)
      { // loop over interseced space_cells
        for (const auto &immersed_cell : immersed_grid)
          {
            typename Triangulation<dim0, spacedim>::cell_iterator
              immersed_cell_t(immersed_cell);
            typename Triangulation<dim1, spacedim>::cell_iterator space_cell_t(
              space_cell);
            const auto test_intersection =
              compute_intersection<dim0, dim1, spacedim>(
                immersed_cell_t, space_cell_t, degree, mapping0, mapping1);
            if (test_intersection.get_points().size() != 0)
              {
                cells_with_quads.push_back(std::make_tuple(immersed_cell_t,
                                                           space_cell_t,
                                                           test_intersection));
              }
          }
      }

    return cells_with_quads;
  }

} // namespace dealii::NonMatching


template std::vector<
  std::tuple<typename dealii::Triangulation<1, 2>::active_cell_iterator,
             typename dealii::Triangulation<2, 2>::active_cell_iterator,
             Quadrature<2>>>
NonMatching::intersect_and_get_quads(
  const GridTools::Cache<1, 2> &immersed_cache,
  const GridTools::Cache<2, 2> &space_cache,
  const unsigned int            degree);

template std::vector<
  std::tuple<typename dealii::Triangulation<2, 2>::active_cell_iterator,
             typename dealii::Triangulation<2, 2>::active_cell_iterator,
             Quadrature<2>>>
NonMatching::intersect_and_get_quads(
  const GridTools::Cache<2, 2> &immersed_cache,
  const GridTools::Cache<2, 2> &space_cache,
  const unsigned int            degree);

template std::vector<
  std::tuple<typename dealii::Triangulation<2, 3>::active_cell_iterator,
             typename dealii::Triangulation<3, 3>::active_cell_iterator,
             Quadrature<3>>>
NonMatching::intersect_and_get_quads(
  const GridTools::Cache<2, 3> &immersed_cache,
  const GridTools::Cache<3, 3> &space_cache,
  const unsigned int            degree);


template std::vector<
  std::tuple<typename dealii::Triangulation<3, 3>::active_cell_iterator,
             typename dealii::Triangulation<3, 3>::active_cell_iterator,
             Quadrature<3>>>
NonMatching::intersect_and_get_quads(
  const GridTools::Cache<3, 3> &immersed_cache,
  const GridTools::Cache<3, 3> &space_cache,
  const unsigned int            degree);
