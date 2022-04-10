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



#ifndef assemble_nitsche_exact_h
#define assemble_nitsche_exact_h

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>

#include <tuple>
#include <vector>

#include "compute_intersections.h"

using namespace dealii;
namespace dealii
{
  namespace NonMatching
  {
    /**
     * @brief Given two non-matching, overlapping grids, this function computes the local contributions
     *        $$M_{ij}:= \int_B \gamma v_i v_j dx$$ exactly, by integrating on
     * the intersection of the two grids. There's no need to change the sparsity
     *        pattern, as the DoFs are all living on the same cell.
     *
     * @tparam dim0 Intrinsic dimension of the first, space grid
     * @tparam dim1 Intrinsic dimension of the second, embedded space
     * @tparam spacedim Ambient space intrinsic dimension
     * @tparam Matrix Matrix type you wish to use
     */
    template <int dim0, int dim1, int spacedim, typename Matrix>
    void
    assemble_nitsche_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &,
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &,
      Matrix &matrix,
      const AffineConstraints<typename Matrix::value_type> &,
      const ComponentMask &,
      const Mapping<dim0, spacedim> &,
      const double penalty = 1.);
  } // namespace NonMatching
} // namespace dealii

#endif
