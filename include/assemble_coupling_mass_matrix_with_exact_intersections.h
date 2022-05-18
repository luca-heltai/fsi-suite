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

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

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
    assemble_coupling_mass_matrix_with_exact_intersections(
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

#ifndef DOXYGEN

#  if defined DEAL_II_WITH_CGAL || defined DEAL_II_WITH_PARMOONOLITH

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
      Assert((dynamic_cast<
                const parallel::distributed::Triangulation<dim1, spacedim> *>(
                &immersed_dh.get_triangulation()) == nullptr),
             ExcMessage("The immersed triangulation can only be a "
                        "parallel::shared::triangulation"));

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



      Table<2, bool> dof_mask(n_dofs_per_space_cell, n_dofs_per_immersed_cell);
      dof_mask.fill(false); // start off by assuming they don't couple

      for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i)
        {
          const auto comp_i = space_fe.system_to_component_index(i).first;
          if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
            {
              for (unsigned int j = 0; j < n_dofs_per_immersed_cell; ++j)
                {
                  const auto comp_j =
                    immersed_fe.system_to_component_index(j).first;
                  if (immersed_gtl[comp_j] == space_gtl[comp_i])
                    {
                      dof_mask(i, j) = true;
                    }
                }
            }
        }

      // const bool dof_mask_is_active =
      //   dof_mask.n_rows() == n_dofs_per_space_cell &&
      //   dof_mask.n_cols() == n_dofs_per_immersed_cell;


      // Loop over vector of tuples, and gather everything together
      for (const auto &infos : cells_and_quads)
        {
          const auto &[first_cell, second_cell, quad_formula] = infos;



          local_cell_matrix = typename Matrix::value_type();

          const unsigned int       n_quad_pts = quad_formula.size();
          const auto &             real_qpts  = quad_formula.get_points();
          std::vector<Point<dim0>> ref_pts_space(n_quad_pts);
          std::vector<Point<dim1>> ref_pts_immersed(n_quad_pts);

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
                  if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
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



          // if (dof_mask_is_active)
          //   {
          //     for (unsigned int i = 0; i < n_dofs_per_space_cell; ++i)
          //       {
          //         const unsigned int comp_i =
          //           space_dh.get_fe().system_to_component_index(i).first;
          //         if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
          //           {
          //             for (unsigned int j = 0; j < n_dofs_per_immersed_cell;
          //                  ++j)
          //               {
          //                 const unsigned int comp_j =
          //                   immersed_dh.get_fe()
          //                     .system_to_component_index(j)
          //                     .first;
          //                 if (space_gtl[comp_i] == immersed_gtl[comp_j])
          //                   {
          //                     space_constraints.distribute_local_to_global(
          //                       local_cell_matrix,
          //                       {local_space_dof_indices[i]},
          //                       immersed_constraints,
          //                       {local_immersed_dof_indices[j]},
          //                       matrix);
          //                   }
          //               }
          //           }
          //       }
          //   }
          // else
          //   {
          space_constraints.distribute_local_to_global(
            local_cell_matrix,
            local_space_dof_indices,
            immersed_constraints,
            local_immersed_dof_indices,
            matrix);
          // }
        }
      matrix.compress(VectorOperation::add);
    }



#  else



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
#  endif
#endif

  } // namespace NonMatching
} // namespace dealii
#endif
