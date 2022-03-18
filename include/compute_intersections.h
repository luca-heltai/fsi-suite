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