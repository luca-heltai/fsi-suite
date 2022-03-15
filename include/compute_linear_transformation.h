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

#ifndef compute_linear_transformation_h
#define compute_linear_transformation_h

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>


/**
 * @brief Given a `dim`-dimensional quadrature formula to integrate over vertices, returns
 *        a `spacedim`-dimensional quadrature formula to integrate in the real
 * space. This function is a generalization of compute_affine_transformation().
 *
 * @tparam dim The template dimension of the original `Quadrature` formula.
 * @tparam spacedim The template dimension of the resulting `Quadrature formula` on the real space
 * @tparam N The number of vertices of the element we need to integrate on.
 * @param quadrature A `Quadrature<dim>` formula
 * @param vertices The `std::array` with `N` vertices you wish to integrate on.
 * @return `Quadrature<spacedim>`object in the real space
 */
template <int dim, int spacedim, int N>
dealii::Quadrature<spacedim>
compute_linear_transformation(
  const dealii::Quadrature<dim> &               quadrature,
  const std::array<dealii::Point<spacedim>, N> &vertices);



// Template implementation
#ifndef DOXYGEN
template <int dim, int spacedim, int N>
dealii::Quadrature<spacedim>
compute_linear_transformation(
  const dealii::Quadrature<dim> &               quadrature,
  const std::array<dealii::Point<spacedim>, N> &vertices)
{
  Assert(N > 1, dealii::ExcInternalError());
  const auto CellType = dealii::ReferenceCell::n_vertices_to_type(
    dim, N); // understand the kind of reference cell from vertices

  dealii::Triangulation<dim, spacedim> tria;
  dealii::GridGenerator::reference_cell(
    tria, CellType); // store reference cell stored in tria
  dealii::FE_Nothing<dim, spacedim> dummy_fe(CellType);
  dealii::DoFHandler<dim, spacedim> dh(tria);
  dh.distribute_dofs(dummy_fe);
  dealii::FEValues<dim, spacedim> fe_values(dummy_fe,
                                            quadrature,
                                            dealii::update_quadrature_points |
                                              dealii::update_JxW_values);
  const auto &                    cell = dh.begin_active();
  for (unsigned int i = 0; i < N; ++i)
    cell->vertex(i) = vertices[i]; // the vertices of this real cell
  fe_values.reinit(cell);

  return dealii::Quadrature<spacedim>(
    fe_values.get_quadrature_points(),
    fe_values.get_JxW_values()); // points and weigths in the real space
}
#endif

#endif