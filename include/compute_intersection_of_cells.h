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


#ifndef compute_intersection_of_cells_h
#define compute_intersection_of_cells_h

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

/**
 * @brief Intersect `cell0` and `cell1` and construct a `Quadrature<spacedim>` of degree `degree``
 *        over the intersection, i.e. in the real space. Mappings for both cells
 * are in `mapping0` and `mapping1`, respectively.
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

namespace dealii::NonMatching{template <int dim0, int dim1, int spacedim>
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
}//namespace NonMatching

#endif