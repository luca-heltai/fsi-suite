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
#ifndef assemble_coupling_exact_h
#define assemble_coupling_exact_h

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <set>
#include <tuple>
#include <vector>

#include "compute_intersections.h"

namespace dealii
{
  namespace NonMatching
  {
    /**
     * @brief Create the coupling mass matrix for non-matching, overlapping grids
     *        in an "exact" way, i.e. by computing the local contributions
     *        $$M_{ij}:= \int_B v_i w_j dx$$ as products of cellwise smooth
              functions on the intersection of the two grids. This information
     is described by a `std::vector<std::tuple>>` where each tuple contains the
     two intersected cells and a Quadrature formula on their intersection.
     *
     * @tparam dim0 Intrinsic dimension of the first, space grid
     * @tparam dim1 Intrinsic dimension of the second, embedded space
     * @tparam spacedim Ambient space intrinsic dimension
     * @tparam Matrix Matrix type you wish to use
     */
    template <int dim0, int dim1, int spacedim, typename Matrix>
    void
    assemble_coupling_exact(
      const dealii::DoFHandler<dim0, spacedim> &,
      const dealii::DoFHandler<dim1, spacedim> &,
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &,
      Matrix &matrix,
      const dealii::AffineConstraints<typename Matrix::value_type> &,
      const dealii::ComponentMask &,
      const dealii::ComponentMask &,
      const dealii::Mapping<dim0, spacedim> &,
      const dealii::Mapping<dim1, spacedim> &,
      const dealii::AffineConstraints<typename Matrix::value_type> &);
  } // namespace NonMatching
} // namespace dealii
#endif
