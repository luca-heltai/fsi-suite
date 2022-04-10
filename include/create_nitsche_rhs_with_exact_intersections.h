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



#ifndef create_exact_rhs_h
#define create_exact_rhs_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <set>
#include <tuple>
#include <vector>

#include "compute_intersections.h"

using namespace dealii;
namespace dealii
{
  namespace NonMatching
  {
    /**
     * @brief Create the r.h.s. $\int_{B} f v$, using exact quadratures on the intersections of the two grids.
     *
     * @tparam dim0
     * @tparam dim1
     * @tparam spacedim
     * @tparam VectorType
     * @param vector
     *
     */
    template <int dim0, int dim1, int spacedim, typename VectorType>
    void
    create_nitsche_rhs_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &,
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &,
      VectorType &vector,
      const AffineConstraints<typename VectorType::value_type> &,
      const Function<spacedim, typename VectorType::value_type> &,
      const Mapping<dim0, spacedim> &,
      const double penalty);
  } // namespace NonMatching
} // namespace dealii

#endif
