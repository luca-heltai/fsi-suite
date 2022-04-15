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


#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>


#ifdef DEAL_II_WITH_CGAL


#  include "assemble_nitsche_with_exact_intersections.h"
using namespace dealii;

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim, typename Matrix>
    void
    assemble_nitsche_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &                    space_dh,
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &                    cells_and_quads,
      Matrix &                                              matrix,
      const AffineConstraints<typename Matrix::value_type> &space_constraints,
      const ComponentMask &                                 space_comps,
      const Mapping<dim0, spacedim> &                       space_mapping,
      const Function<spacedim, typename Matrix::value_type>
        &          nitsche_coefficient,
      const double penalty)
    {
      AssertDimension(matrix.m(), space_dh.n_dofs());
      AssertDimension(matrix.n(), space_dh.n_dofs());
      Assert(dim1 <= dim0,
             ExcMessage("This function can only work if dim1<=dim0"));



      const auto &space_fe = space_dh.get_fe();

      const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();

      const unsigned int n_space_fe_components = space_fe.n_components();

      FullMatrix<double> local_cell_matrix(n_dofs_per_space_cell,
                                           n_dofs_per_space_cell);
      // DoF indices
      std::vector<types::global_dof_index> local_space_dof_indices(
        n_dofs_per_space_cell);


      const ComponentMask space_c =
        (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true) :
                                   space_comps);


      AssertDimension(space_c.size(), n_space_fe_components);

      std::vector<unsigned int> space_gtl(n_space_fe_components,
                                          numbers::invalid_unsigned_int);

      for (unsigned int i = 0, j = 0; i < n_space_fe_components; ++i)
        {
          if (space_c[i])
            space_gtl[i] = j++;
        }



      // Loop over vector of tuples, and gather everything together
      double h;
      for (const auto &infos : cells_and_quads)
        {
          const auto &[first_cell, second_cell, quad_formula] = infos;
          if (first_cell->is_active())
            {
              local_cell_matrix = typename Matrix::value_type();

              const unsigned int  n_quad_pts = quad_formula.size();
              const auto &        real_qpts  = quad_formula.get_points();
              std::vector<double> nitsche_coefficient_values(n_quad_pts);
              nitsche_coefficient.value_list(real_qpts,
                                             nitsche_coefficient_values);

              std::vector<Point<spacedim>> ref_pts_space(n_quad_pts);

              space_mapping.transform_points_real_to_unit_cell(first_cell,
                                                               real_qpts,
                                                               ref_pts_space);

              h               = first_cell->diameter();
              const auto &JxW = quad_formula.get_weights();
              for (unsigned int q = 0; q < n_quad_pts; ++q)
                {
                  const auto &q_ref_point = ref_pts_space[q];
                  for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i)
                    {
                      const unsigned int comp_i =
                        space_dh.get_fe().system_to_component_index(i).first;
                      if (comp_i != numbers::invalid_unsigned_int)
                        {
                          for (unsigned int j = 0; j < n_dofs_per_space_cell;
                               ++j)
                            {
                              const unsigned int comp_j =
                                space_dh.get_fe()
                                  .system_to_component_index(j)
                                  .first;
                              if (space_gtl[comp_i] == space_gtl[comp_j])
                                {
                                  local_cell_matrix(i, j) +=
                                    nitsche_coefficient_values[q] *
                                    (penalty / h) *
                                    space_fe.shape_value(i, q_ref_point) *
                                    space_fe.shape_value(j, q_ref_point) *
                                    JxW[q];
                                }
                            }
                        }
                    }
                }
              typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
                *first_cell, &space_dh);

              space_cell_dh->get_dof_indices(local_space_dof_indices);
              space_constraints.distribute_local_to_global(
                local_cell_matrix, local_space_dof_indices, matrix);
            }
        }
    }

  } // namespace NonMatching
} // namespace dealii



#else



template <int dim0, int dim1, int spacedim, typename Matrix>
void
dealii::NonMatching::assemble_nitsche_with_exact_intersections(
  const DoFHandler<dim0, spacedim> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
               typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
               dealii::Quadrature<spacedim>>> &,
  Matrix &,
  const AffineConstraints<typename Matrix::value_type> &,
  const ComponentMask &,
  const Mapping<dim0, spacedim> &)
{
  Assert(false,
         ExcMessage("This function needs CGAL to be installed, "
                    "but cmake could not find it."));
  return {};
}


#endif



template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<1, 1, 1>(
  const DoFHandler<1, 1> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<1, 1>::cell_iterator,
               typename dealii::Triangulation<1, 1>::cell_iterator,
               dealii::Quadrature<1>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<1, 1> &,
  const Function<1, double> &nitsche_coefficient,
  const double);

template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<2, 1, 2>(
  const DoFHandler<2, 2> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
               typename dealii::Triangulation<1, 2>::cell_iterator,
               dealii::Quadrature<2>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<2, 2> &,
  const Function<2, double> &nitsche_coefficient,
  const double);


template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<2, 2, 2>(
  const DoFHandler<2, 2> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
               typename dealii::Triangulation<2, 2>::cell_iterator,
               dealii::Quadrature<2>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<2, 2> &,
  const Function<2, double> &nitsche_coefficient,
  const double);


template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<3, 1, 3>(
  const DoFHandler<3, 3> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
               typename dealii::Triangulation<1, 3>::cell_iterator,
               dealii::Quadrature<3>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<3, 3> &,
  const Function<3, double> &nitsche_coefficient,
  const double);

template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<3, 2, 3>(
  const DoFHandler<3, 3> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
               typename dealii::Triangulation<2, 3>::cell_iterator,
               dealii::Quadrature<3>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<3, 3> &,
  const Function<3, double> &nitsche_coefficient,
  const double);



template void
dealii::NonMatching::assemble_nitsche_with_exact_intersections<3, 3, 3>(
  const DoFHandler<3, 3> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
               typename dealii::Triangulation<3, 3>::cell_iterator,
               dealii::Quadrature<3>>> &,
  dealii::SparseMatrix<double> &,
  const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
  const ComponentMask &,
  const Mapping<3, 3> &,
  const Function<3, double> &nitsche_coefficient,
  const double);
