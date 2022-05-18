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

#ifndef create_coupling_sparsity_pattern_with_exact_intersections_h
#define create_coupling_sparsity_pattern_with_exact_intersections_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <boost/geometry.hpp>

namespace dealii
{
  namespace NonMatching
  {
    /**
     * @brief Create a coupling sparsity pattern of two non-matching, overlapped
     *        grids. As it relies on `compute_intersection`, the "small"
     *        intersections do not enter in the sparsity pattern.
     * @param intersections_info A vector of tuples where the i-th entry
     * contains two `active_cell_iterator`s to the intersected cells
     * @param space_dh `DoFHandler` object for the space grid
     * @param immersed_dh `DoFHandler` object for the embedded grid
     * @param sparsity The sparsity pattern to be filled
     * @param constraints `AffineConstraints` for the space grid
     * @param space_comps Mask for the space space components of the finite
     * element
     * @param immersed_comps Mask for the embedded components of the finite
     * element
     * @param immersed_constraints `AffineConstraints` for the embedded grid
     *
     *
     */
    template <int dim0,
              int dim1,
              int spacedim,
              typename Sparsity,
              typename number = double>
    void
    create_coupling_sparsity_pattern_with_exact_intersections(
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &intersections_info,
      const DoFHandler<dim0, spacedim> &space_dh,
      const DoFHandler<dim1, spacedim> &immersed_dh,
      Sparsity &                        sparsity,
      const AffineConstraints<number> & constraints =
        AffineConstraints<number>(),
      const ComponentMask &            space_comps    = ComponentMask(),
      const ComponentMask &            immersed_comps = ComponentMask(),
      const AffineConstraints<number> &immersed_constraints =
        AffineConstraints<number>());

#ifndef DOXYGEN

#  if defined DEAL_II_WITH_CGAL || defined DEAL_II_WITH_PARMOONOLITH

    using namespace dealii;
    template <int dim0,
              int dim1,
              int spacedim,
              typename Sparsity,
              typename number>
    void
    create_coupling_sparsity_pattern_with_exact_intersections(
      const std::vector<std::tuple<
        typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>> &intersections_info,
      const DoFHandler<dim0, spacedim> &space_dh,
      const DoFHandler<dim1, spacedim> &immersed_dh,
      Sparsity &                        sparsity,
      const AffineConstraints<number> & constraints,
      const ComponentMask &             space_comps,
      const ComponentMask &             immersed_comps,
      const AffineConstraints<number> & immersed_constraints)
    {
      AssertDimension(sparsity.n_rows(), space_dh.n_dofs());
      AssertDimension(sparsity.n_cols(), immersed_dh.n_dofs());
      Assert(dim1 <= dim0,
             ExcMessage("This function can only work if dim1 <= dim0"));
      Assert((dynamic_cast<
                const parallel::distributed::Triangulation<dim1, spacedim> *>(
                &immersed_dh.get_triangulation()) == nullptr),
             ExcNotImplemented());



      const auto &       space_fe              = space_dh.get_fe();
      const auto &       immersed_fe           = immersed_dh.get_fe();
      const unsigned int n_dofs_per_space_cell = space_fe.n_dofs_per_cell();
      const unsigned int n_dofs_per_immersed_cell =
        immersed_fe.n_dofs_per_cell();
      const unsigned int n_space_fe_components    = space_fe.n_components();
      const unsigned int n_immersed_fe_components = immersed_fe.n_components();
      std::vector<types::global_dof_index> space_dofs(n_dofs_per_space_cell);
      std::vector<types::global_dof_index> immersed_dofs(
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


      // Global 2 Local indices
      std::vector<unsigned int> space_gtl(n_space_fe_components);
      std::vector<unsigned int> immersed_gtl(n_immersed_fe_components);
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

      const bool dof_mask_is_active =
        dof_mask.n_rows() == n_dofs_per_space_cell &&
        dof_mask.n_cols() == n_dofs_per_immersed_cell;

      // Whenever the BB space_cell intersects the BB of an embedded cell,
      // those DoFs have to be recorded

      for (const auto &it : intersections_info)
        {
          const auto &space_cell    = std::get<0>(it);
          const auto &immersed_cell = std::get<1>(it);
          typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
            *space_cell, &space_dh);
          typename DoFHandler<dim1, spacedim>::cell_iterator immersed_cell_dh(
            *immersed_cell, &immersed_dh);

          space_cell_dh->get_dof_indices(space_dofs);
          immersed_cell_dh->get_dof_indices(immersed_dofs);

          if (dof_mask_is_active)
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
                              // local_cell_matrix(i, j) +=
                              constraints.add_entries_local_to_global(
                                {space_dofs[i]},
                                immersed_constraints,
                                {immersed_dofs[j]},
                                sparsity,
                                true);
                            }
                        }
                    }
                }
            }
          else
            {
              constraints.add_entries_local_to_global(space_dofs,
                                                      immersed_constraints,
                                                      immersed_dofs,
                                                      sparsity,
                                                      true,
                                                      dof_mask);
            }
        }
    }

#  else

    namespace dealii
    {
      namespace NonMatching
      {
        template <int dim0,
                  int dim1,
                  int spacedim,
                  typename Sparsity,
                  typename number>
        void

        create_coupling_sparsity_pattern_with_exact_intersections(
          const std::vector<std::tuple<
            typename dealii::Triangulation<dim0, spacedim>::cell_iterator,
            typename dealii::Triangulation<dim1, spacedim>::cell_iterator,
            dealii::Quadrature<spacedim>>> &,
          const DoFHandler<dim0, spacedim> &,
          const DoFHandler<dim1, spacedim> &,
          Sparsity &,
          const AffineConstraints<number> &,
          const ComponentMask &,
          const ComponentMask &,
          const AffineConstraints<number> &)
        {
          Assert(false,
                 ExcMessage("This function needs CGAL to be installed, "
                            "but cmake could not find it."));
        }

#  endif

#endif

  } // namespace NonMatching
} // namespace dealii

#endif