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

#include "assemble_coupling_mass_matrix_with_exact_intersections.h"

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include "lac.h"

using namespace dealii;

namespace dealii
{
  namespace NonMatching
  {
#if defined DEAL_II_WITH_CGAL || defined DEAL_II_WITH_PARMOONOLITH

    template <int dim0, int dim1, int spacedim, typename Matrix>
    void
    assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &space_dh,
      const DoFHandler<dim1, spacedim> &immersed_dh,
      const std::vector<
        std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                   typename Triangulation<dim1, spacedim>::cell_iterator,
                   Quadrature<spacedim>>> &                 cells_and_quads,
      Matrix &                                              matrix,
      const AffineConstraints<typename Matrix::value_type> &space_constraints,
      const ComponentMask &                                 space_comps,
      const ComponentMask &                                 immersed_comps,
      const Mapping<dim0, spacedim> &                       space_mapping,
      const Mapping<dim1, spacedim> &                       immersed_mapping,
      const AffineConstraints<typename Matrix::value_type>
        &immersed_constraints)
    {
      AssertDimension(matrix.m(), space_dh.n_dofs());
      AssertDimension(matrix.n(), immersed_dh.n_dofs());
      Assert(dim1 <= dim0,
             ExcMessage("This function can only work if dim1<=dim0"));
      // Assert((dynamic_cast<
      //           const parallel::distributed::Triangulation<dim1, spacedim>
      //           *>( &immersed_dh.get_triangulation()) == nullptr),
      //        ExcNotImplemented());



      const auto &space_fe    = space_dh.get_fe();
      const auto &immersed_fe = immersed_dh.get_fe();

      const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
      const unsigned int n_dofs_per_immersed_cell =
        immersed_fe.n_dofs_per_cell();

      const unsigned int n_space_fe_components    = space_fe.n_components();
      const unsigned int n_immersed_fe_components = immersed_fe.n_components();

      FullMatrix<double> local_cell_matrix(n_dofs_per_space_cell,
                                           n_dofs_per_immersed_cell);
      // DoF indices
      std::vector<types::global_dof_index> local_space_dof_indices(
        n_dofs_per_space_cell);
      std::vector<types::global_dof_index> local_immersed_dof_indices(
        n_dofs_per_immersed_cell);

      const ComponentMask space_c =
        (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true) :
                                   space_comps);
      const ComponentMask immersed_c =
        (immersed_comps.size() == 0 ?
           ComponentMask(n_immersed_fe_components, true) :
           immersed_comps);

      AssertDimension(space_c.size(), n_space_fe_components);
      AssertDimension(immersed_c.size(), n_immersed_fe_components);

      std::vector<unsigned int> space_gtl(n_space_fe_components,
                                          numbers::invalid_unsigned_int);
      std::vector<unsigned int> immersed_gtl(n_immersed_fe_components,
                                             numbers::invalid_unsigned_int);
      for (unsigned int i = 0, j = 0; i < n_space_fe_components; ++i)
        {
          if (space_c[i])
            space_gtl[i] = j++;
        }


      for (unsigned int i = 0, j = 0; i < n_immersed_fe_components; ++i)
        {
          if (immersed_c[i])
            immersed_gtl[i] = j++;
        }



      // Loop over vector of tuples, and gather everything together

      for (const auto &infos : cells_and_quads)
        {
          const auto &[first_cell, second_cell, quad_formula] = infos;



          local_cell_matrix = typename Matrix::value_type();

          const unsigned int           n_quad_pts = quad_formula.size();
          const auto &                 real_qpts  = quad_formula.get_points();
          std::vector<Point<spacedim>> ref_pts_space(n_quad_pts);
          std::vector<Point<dim1>>     ref_pts_immersed(n_quad_pts);

          space_mapping.transform_points_real_to_unit_cell(first_cell,
                                                           real_qpts,
                                                           ref_pts_space);
          immersed_mapping.transform_points_real_to_unit_cell(second_cell,
                                                              real_qpts,
                                                              ref_pts_immersed);
          const auto &JxW = quad_formula.get_weights();
          for (unsigned int q = 0; q < n_quad_pts; ++q)
            {
              for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i)
                {
                  const unsigned int comp_i =
                    space_dh.get_fe().system_to_component_index(i).first;
                  if (comp_i != numbers::invalid_unsigned_int)
                    {
                      for (unsigned int j = 0; j < n_dofs_per_immersed_cell;
                           ++j)
                        {
                          const unsigned int comp_j =
                            immersed_dh.get_fe()
                              .system_to_component_index(j)
                              .first;
                          if (space_gtl[comp_i] == immersed_gtl[comp_j])
                            {
                              local_cell_matrix(i, j) +=
                                space_fe.shape_value(i, ref_pts_space[q]) *
                                immersed_fe.shape_value(j,
                                                        ref_pts_immersed[q]) *
                                JxW[q];
                            }
                        }
                    }
                }
            }
          typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
            *first_cell, &space_dh);
          typename DoFHandler<dim1, spacedim>::cell_iterator immersed_cell_dh(
            *second_cell, &immersed_dh);
          space_cell_dh->get_dof_indices(local_space_dof_indices);
          immersed_cell_dh->get_dof_indices(local_immersed_dof_indices);



          space_constraints.distribute_local_to_global(
            local_cell_matrix,
            local_space_dof_indices,
            immersed_constraints,
            local_immersed_dof_indices,
            matrix);
        }
    }



#else



    template <int dim0, int dim1, int spacedim, typename Matrix>
    void
    assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<dim0, spacedim> &,
      const DoFHandler<dim1, spacedim> &,
      const std::vector<
        std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                   typename Triangulation<dim1, spacedim>::cell_iterator,
                   Quadrature<spacedim>>> &,
      Matrix &,
      const AffineConstraints<typename Matrix::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<dim0, spacedim> &,
      const Mapping<dim1, spacedim> &,
      const AffineConstraints<typename Matrix::value_type> &)
    {
      Assert(false,
             ExcMessage(
               "This function needs CGAL or PARMOONOLITH to be installed, "
               "but cmake could not either."));
    }



#endif



    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<1, 1> &,
      const DoFHandler<1, 1> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<1, 1>::cell_iterator,
                   typename dealii::Triangulation<1, 1>::cell_iterator,
                   dealii::Quadrature<1>>> &,
      dealii::SparseMatrix<double> &,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<1, 1> &space_mapping,
      const Mapping<1, 1> &immersed_mapping,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<2, 2> &,
      const DoFHandler<1, 2> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                   typename dealii::Triangulation<1, 2>::cell_iterator,
                   dealii::Quadrature<2>>> &,
      dealii::SparseMatrix<double> &,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<2, 2> &space_mapping,
      const Mapping<1, 2> &immersed_mapping,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<2, 2> &,
      const DoFHandler<2, 2> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                   typename dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Quadrature<2>>> &,
      dealii::SparseMatrix<double> &,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<2, 2> &space_mapping,
      const Mapping<2, 2> &immersed_mapping,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &);



    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<3, 3> &,
      const DoFHandler<1, 3> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
                   typename dealii::Triangulation<1, 3>::cell_iterator,
                   dealii::Quadrature<3>>> &,
      dealii::SparseMatrix<double> &,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<3, 3> &space_mapping,
      const Mapping<1, 3> &immersed_mapping,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<3, 3> &,
      const DoFHandler<2, 3> &,
      const std::vector<
        std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
                   typename dealii::Triangulation<2, 3>::cell_iterator,
                   dealii::Quadrature<3>>> &,
      dealii::SparseMatrix<double> &,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<3, 3> &space_mapping,
      const Mapping<2, 3> &immersed_mapping,
      const AffineConstraints<dealii::SparseMatrix<double>::value_type> &);

    template void
    assemble_coupling_mass_matrix_with_exact_intersections(
      const DoFHandler<3, 3> &,
      const DoFHandler<3, 3> &,
      const std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                   typename Triangulation<3, 3>::cell_iterator,
                                   Quadrature<3>>> &,
      SparseMatrix<double> &,
      const AffineConstraints<typename SparseMatrix<double>::value_type> &,
      const ComponentMask &,
      const ComponentMask &,
      const Mapping<3, 3> &space_mapping,
      const Mapping<3, 3> &immersed_mapping,
      const AffineConstraints<typename SparseMatrix<double>::value_type> &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      2,
      1,
      2,
      dealii::TrilinosWrappers::SparseMatrix>(
      dealii::DoFHandler<2, 2> const &,
      dealii::DoFHandler<1, 2> const &,
      std::vector<
        std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Triangulation<1, 2>::cell_iterator,
                   dealii::Quadrature<2>>,
        std::allocator<std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Triangulation<1, 2>::cell_iterator,
                                  dealii::Quadrature<2>>>> const &,
      dealii::TrilinosWrappers::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<2, 2> const &,
      dealii::Mapping<1, 2> const &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      2,
      2,
      2,
      dealii::TrilinosWrappers::SparseMatrix>(
      dealii::DoFHandler<2, 2> const &,
      dealii::DoFHandler<2, 2> const &,
      std::vector<
        std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Quadrature<2>>,
        std::allocator<std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Quadrature<2>>>> const &,
      dealii::TrilinosWrappers::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<2, 2> const &,
      dealii::Mapping<2, 2> const &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      3,
      2,
      3,
      dealii::TrilinosWrappers::SparseMatrix>(
      dealii::DoFHandler<3, 3> const &,
      dealii::DoFHandler<2, 3> const &,
      std::vector<
        std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Triangulation<2, 3>::cell_iterator,
                   dealii::Quadrature<3>>,
        std::allocator<std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Triangulation<2, 3>::cell_iterator,
                                  dealii::Quadrature<3>>>> const &,
      dealii::TrilinosWrappers::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<3, 3> const &,
      dealii::Mapping<2, 3> const &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      3,
      3,
      3,
      dealii::TrilinosWrappers::SparseMatrix>(
      dealii::DoFHandler<3, 3> const &,
      dealii::DoFHandler<3, 3> const &,
      std::vector<
        std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Quadrature<3>>,
        std::allocator<std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Quadrature<3>>>> const &,
      dealii::TrilinosWrappers::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<3, 3> const &,
      dealii::Mapping<3, 3> const &,
      dealii::AffineConstraints<
        dealii::TrilinosWrappers::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      2,
      1,
      2,
      dealii::PETScWrappers::MPI::SparseMatrix>(
      dealii::DoFHandler<2, 2> const &,
      dealii::DoFHandler<1, 2> const &,
      std::vector<
        std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Triangulation<1, 2>::cell_iterator,
                   dealii::Quadrature<2>>,
        std::allocator<std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Triangulation<1, 2>::cell_iterator,
                                  dealii::Quadrature<2>>>> const &,
      dealii::PETScWrappers::MPI::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<2, 2> const &,
      dealii::Mapping<1, 2> const &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      2,
      2,
      2,
      dealii::PETScWrappers::MPI::SparseMatrix>(
      dealii::DoFHandler<2, 2> const &,
      dealii::DoFHandler<2, 2> const &,
      std::vector<
        std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Triangulation<2, 2>::cell_iterator,
                   dealii::Quadrature<2>>,
        std::allocator<std::tuple<dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Triangulation<2, 2>::cell_iterator,
                                  dealii::Quadrature<2>>>> const &,
      dealii::PETScWrappers::MPI::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<2, 2> const &,
      dealii::Mapping<2, 2> const &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      3,
      2,
      3,
      dealii::PETScWrappers::MPI::SparseMatrix>(
      dealii::DoFHandler<3, 3> const &,
      dealii::DoFHandler<2, 3> const &,
      std::vector<
        std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Triangulation<2, 3>::cell_iterator,
                   dealii::Quadrature<3>>,
        std::allocator<std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Triangulation<2, 3>::cell_iterator,
                                  dealii::Quadrature<3>>>> const &,
      dealii::PETScWrappers::MPI::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<3, 3> const &,
      dealii::Mapping<2, 3> const &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &);

    template void
    dealii::NonMatching::assemble_coupling_mass_matrix_with_exact_intersections<
      3,
      3,
      3,
      dealii::PETScWrappers::MPI::SparseMatrix>(
      dealii::DoFHandler<3, 3> const &,
      dealii::DoFHandler<3, 3> const &,
      std::vector<
        std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Triangulation<3, 3>::cell_iterator,
                   dealii::Quadrature<3>>,
        std::allocator<std::tuple<dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Triangulation<3, 3>::cell_iterator,
                                  dealii::Quadrature<3>>>> const &,
      dealii::PETScWrappers::MPI::SparseMatrix &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &,
      dealii::ComponentMask const &,
      dealii::ComponentMask const &,
      dealii::Mapping<3, 3> const &,
      dealii::Mapping<3, 3> const &,
      dealii::AffineConstraints<
        dealii::PETScWrappers::MPI::SparseMatrix::value_type> const &);
  } // namespace NonMatching
} // namespace dealii