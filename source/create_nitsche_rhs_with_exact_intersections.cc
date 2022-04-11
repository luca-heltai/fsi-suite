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



#ifdef DEAL_II_WITH_CGAL

#  include "create_nitsche_rhs_with_exact_intersections.h"

using namespace dealii;

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim, typename VectorType>
    void
    create_nitsche_rhs_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &space_dh,
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &cells_and_quads,
      VectorType &                      rhs,
      const AffineConstraints<typename VectorType::value_type>
        &                            space_constraints,
      const Mapping<dim0, spacedim> &space_mapping,
      const Function<spacedim, typename VectorType::value_type> &rhs_function,
      const Function<spacedim, typename VectorType::value_type> &coefficient,
      const double                                               penalty)
    {
      AssertDimension(rhs.size(), space_dh.n_dofs());
      Assert(dim1 <= dim0,
             ExcMessage("This function can only work if dim1<=dim0"));


      const auto &space_fe = space_dh.get_fe();

      const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();



      Vector<double> local_rhs(n_dofs_per_space_cell);
      // DoF indices
      std::vector<types::global_dof_index> local_space_dof_indices(
        n_dofs_per_space_cell);



      // Loop over vector of tuples, and gather everything together

      double h;
      for (const auto &infos : cells_and_quads)
        {
          const auto &[first_cell, second_cell, quad_formula] = infos;

          if (first_cell->is_active())
            {
              h         = first_cell->diameter();
              local_rhs = typename VectorType::value_type();


              const unsigned int n_quad_pts = quad_formula.size();
              const auto &       real_qpts  = quad_formula.get_points();
              std::vector<Point<spacedim>> ref_pts_space(n_quad_pts);
              std::vector<double>          rhs_function_values(n_quad_pts);
              rhs_function.value_list(real_qpts, rhs_function_values);


              std::vector<double> coefficient_values(n_quad_pts);
              coefficient.value_list(real_qpts, coefficient_values);

              space_mapping.transform_points_real_to_unit_cell(first_cell,
                                                               real_qpts,
                                                               ref_pts_space);

              const auto &JxW = quad_formula.get_weights();
              for (unsigned int q = 0; q < n_quad_pts; ++q)
                {
                  const auto &q_ref_point = ref_pts_space[q];
                  for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i)
                    {
                      local_rhs(i) += coefficient_values[q] * (penalty / h) *
                                      space_fe.shape_value(i, q_ref_point) *
                                      rhs_function_values[q] * JxW[q];
                    }
                }
              typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
                *first_cell, &space_dh);

              space_cell_dh->get_dof_indices(local_space_dof_indices);
              space_constraints.distribute_local_to_global(
                local_rhs, local_space_dof_indices, rhs);
            }
        }
    }



#else



template <int dim0, int dim1, int spacedim, typename VectorType>
void
create_nitsche_rhs_with_exact_intersections(
  const DoFHandler<dim0, spacedim> &,
  const std::vector<
    std::tuple<typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
               typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
               dealii::Quadrature<spacedim>>> &,
  VectorType &vector,
  const AffineConstraints<typename VectorType::value_type> &,
  const Function<spacedim, typename VectorType::value_type> &,
  const Mapping<dim0, spacedim> &);
{
  Assert(false,
         ExcMessage("This function needs CGAL to be installed, "
                    "but cmake could not find it."));
  return {};
}


#endif



    template void
    create_nitsche_rhs_with_exact_intersections<2, 2, 2>(
      const DoFHandler<2, 2> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                   typename dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Quadrature<2>>> &,
      Vector<double> &vector,
      const AffineConstraints<double> &,
      const Mapping<2, 2> &,
      const Function<2, double> &,
      const Function<2, double> &,
      const double);



    template void
    create_nitsche_rhs_with_exact_intersections<2, 1, 2>(
      const DoFHandler<2, 2> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                   typename dealii::Triangulation<1, 2>::cell_iterator,
                   dealii::Quadrature<2>>> &,
      Vector<double> &vector,
      const AffineConstraints<double> &,
      const Mapping<2, 2> &,
      const Function<2, double> &,
      const Function<2, double> &,
      const double);


    template void
    create_nitsche_rhs_with_exact_intersections<3, 3, 3>(
      const DoFHandler<3, 3> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
                   typename dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Quadrature<3>>> &,
      Vector<double> &vector,
      const AffineConstraints<double> &,
      const Mapping<3, 3> &,
      const Function<3, double> &,
      const Function<3, double> &,
      const double);


  } // namespace NonMatching
} // namespace dealii
