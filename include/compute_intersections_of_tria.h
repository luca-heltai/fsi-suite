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


#ifndef compute_intersections_and_quads_h
#define compute_intersections_and_quads_h

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <set>
#include <tuple>
#include <vector>

#include "compute_intersection_of_cells.h"



namespace dealii::NonMatching
{
  template <int dim0, int dim1, int spacedim>
  std::vector<std::tuple<
    typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
    typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
    dealii::Quadrature<spacedim>>>
  intersect_and_get_quads(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const unsigned int                      degree);

}



#endif