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

#ifndef create_coupling_sparsity_pattern_with_exact_intersections_h
#define create_coupling_sparsity_pattern_with_exact_intersections_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <boost/geometry.hpp>

namespace dealii
{
  namespace NonMatching
  {
    /**
     * @brief Create a coupling sparsity pattern of two non-matching, overlapped
     *        grids. As it relies on `compute_intersection`, the "small"
     *        intersections do not enter in the sparsity pattern.
     * @param intersections_info A vector of tuples where the i-th entry
     * contains two `active_cell_iterator`s to the intersected cells
     * @param space_dh `DoFHandler` object for the space grid
     * @param immersed_dh `DoFHandler` object for the embedded grid
     * @param sparsity The sparsity pattern to be filled
     * @param constraints `AffineConstraints` for the space grid
     * @param space_comps Mask for the space space components of the finite
     * element
     * @param immersed_comps Mask for the embedded components of the finite
     * element
     * @param immersed_constraints `AffineConstraints` for the embedded grid
     *
     *
     */
    template <int dim0,
              int dim1,
              int spacedim,
              typename Sparsity,
              typename number = double>
    void
    create_coupling_sparsity_pattern_with_exact_intersections(
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &intersections_info,
      const DoFHandler<dim0, spacedim> &space_dh,
      const DoFHandler<dim1, spacedim> &immersed_dh,
      Sparsity                         &sparsity,
      const AffineConstraints<number>  &constraints =
        AffineConstraints<number>(),
      const ComponentMask             &space_comps    = ComponentMask(),
      const ComponentMask             &immersed_comps = ComponentMask(),
      const AffineConstraints<number> &immersed_constraints =
        AffineConstraints<number>());

  } // namespace NonMatching
} // namespace dealii

#endif