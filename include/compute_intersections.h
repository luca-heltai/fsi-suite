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



#ifndef compute_intersections_h
#define compute_intersections_h

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <set>
#include <tuple>
#include <vector>


namespace dealii::NonMatching
{
  /**
   * @brief Intersect `cell0` and `cell1` and construct a `Quadrature<spacedim>` of degree `degree``
   *        over the intersection, i.e. in the real space. Mappings for both
   * cells are in `mapping0` and `mapping1`, respectively.
   *
   * @tparam dim0
   * @tparam dim1
   * @tparam spacedim
   * @param cell0 A `cell_iteratator` to the first cell
   * @param cell1 A `cell_iteratator` to the first cell
   * @param degree The degree of the `Quadrature` you want to build there
   * @param mapping0 The `Mapping` object describing the first cell
   * @param mapping1 The `Mapping` object describing the second cell
   * @return Quadrature<spacedim>
   */
  template <int dim0, int dim1, int spacedim>
  dealii::Quadrature<spacedim>
  compute_intersection(
    const typename dealii::Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename dealii::Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                                   degree,
    const dealii::Mapping<dim0, spacedim> &mapping0 =
      (dealii::ReferenceCells::get_hypercube<dim0>()
         .template get_default_linear_mapping<dim0, spacedim>()),
    const dealii::Mapping<dim1, spacedim> &mapping1 =
      (dealii::ReferenceCells::get_hypercube<dim1>()
         .template get_default_linear_mapping<dim1, spacedim>()));



  /**
   * @brief Given two triangulations cached inside `GridTools::Cache` objects, compute all intersections between the two
   * and return a vector where each entry is a tuple containing iterators to the
   * respective cells and a `Quadrature<spacedim>` formula to integrate over the
   * intersection.
   *
   * @tparam dim0 Intrinsic dimension of the immersed grid
   * @tparam dim1 Intrinsic dimension of the ambient grid
   * @tparam spacedim
   * @param space_cache
   * @param immersed_cache
   * @param degree Degree of the desired quadrature formula
   * @return std::vector<std::tuple<
   * typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
   * typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
   * dealii::Quadrature<spacedim>>>
   */
  template <int dim0, int dim1, int spacedim>
  std::vector<std::tuple<
    typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
    typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
    dealii::Quadrature<spacedim>>>
  compute_intersection(const GridTools::Cache<dim0, spacedim> &space_cache,
                       const GridTools::Cache<dim1, spacedim> &immersed_cache,
                       const unsigned int                      degree);



} // namespace dealii::NonMatching

#endif