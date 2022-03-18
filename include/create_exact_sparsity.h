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

#ifndef create_exact_sparsity_h
#define create_exact_sparsity_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <boost/geometry.hpp>



using namespace dealii;
namespace dealii::NonMatching
{
  template <int dim0,
            int dim1,
            int spacedim,
            typename Sparsity,
            typename number = double>
  void
  create_exact_sparsity_pattern(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const DoFHandler<dim0, spacedim> &      space_dh,
    const DoFHandler<dim1, spacedim> &      immersed_dh,
    Sparsity &                              sparsity,
    const AffineConstraints<number> &       constraints,
    const ComponentMask &                   space_comps,
    const ComponentMask &                   immersed_comps,
    const AffineConstraints<number> &       immersed_constraint);

}



#endif