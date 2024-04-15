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

#include "pdes/lagrange_multipliers.h"

#include <deal.II/base/logstream.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "lac_initializer.h"

using namespace dealii;

namespace PDEs
{
  template <int spacedim, typename LacType>
  LagrangeMultipliers<spacedim, LacType>::LagrangeMultipliers()
    : ParameterAcceptor("Lagrange Multipliers")
    , space("u", "Space")
    , space_cache(space.triangulation)
    , embedded("w", "Embedded")
    , embedded_cache(embedded.triangulation)
    , coupling("/Coupling")
    , mass_solver("/Mass solver")
  {
    add_parameter("System solver type",
                  system_solver_type,
                  "The algorithm use to solve the system. This can be either "
                  "schur, direct, diagonal, back, or forward.",
                  this->prm,
                  Patterns::Selection("schur|direct|diagonal|back|forward"));
    add_parameter("Filter space averages",
                  filter_space_averages,
                  "Filter constant modes in the space AMG preconditioner ");
    // add_parameter("Filter embedded averages",
    //               filter_embedded_averages,
    //               "Filter constant modes in the embedded AMG preconditioner
    //               ");
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::generate_grids()
  {
    TimerOutput::Scope timer_section(space.timer, "generate_grids");
    space.grid_generator.generate(space.triangulation);
    embedded.grid_generator.generate(embedded.triangulation);
    coupling.initialize(space_cache,
                        space.dof_handler,
                        space.constraints,
                        embedded_cache,
                        embedded.dof_handler,
                        embedded.constraints);
    coupling.adjust_grid_refinements(space.triangulation,
                                     embedded.triangulation,
                                     true);
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::setup_system()
  {
    space.setup_system();
    embedded.setup_system();

    const auto row_indices = space.dof_handler.locally_owned_dofs();
    const auto col_indices = embedded.dof_handler.locally_owned_dofs();

    LAC::Initializer init(row_indices,
                          IndexSet(),
                          space.mpi_communicator,
                          col_indices);

    DynamicSparsityPattern dsp(space.dof_handler.n_dofs(),
                               embedded.dof_handler.n_dofs(),
                               row_indices);

    coupling.assemble_sparsity(dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               row_indices,
                                               space.mpi_communicator,
                                               space.locally_relevant_dofs[0]);

    coupling_matrix.reinit(row_indices,
                           col_indices,
                           dsp,
                           space.mpi_communicator);
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::assemble_system()
  {
    {
      TimerOutput::Scope timer_section(space.timer,
                                       "Assemble stiffness system");
      // Stiffness matrix and rhs
      typename LinearProblem<spacedim, spacedim, LacType>::ScratchData scratch(
        *space.mapping,
        space.finite_element(),
        space.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<spacedim, spacedim, LacType>::CopyData copy(
        space.finite_element().n_dofs_per_cell());

      for (const auto &cell : space.dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto &cell_matrix     = copy.matrices[0];
            auto &cell_rhs        = copy.vectors[0];
            cell_matrix           = 0;
            cell_rhs              = 0;
            const auto &fe_values = scratch.reinit(cell);
            cell->get_dof_indices(copy.local_dof_indices[0]);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  {
                    for (const unsigned int j : fe_values.dof_indices())
                      cell_matrix(i, j) +=
                        (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                         fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                         fe_values.JxW(q_index));           // dx
                    cell_rhs(i) +=
                      (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                       space.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                  }
              }

            space.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              space.matrix,
              space.rhs);
          }

      space.matrix.compress(VectorOperation::add);
      space.rhs.compress(VectorOperation::add);
    }
    {
      TimerOutput::Scope timer_section(space.timer, "Assemble coupling system");
      coupling_matrix = 0.0;
      coupling.assemble_matrix(coupling_matrix);
      coupling_matrix.compress(VectorOperation::add);
    }
    {
      // Embedded mass matrix and rhs
      typename LinearProblem<dim, spacedim, LacType>::ScratchData scratch(
        *embedded.mapping,
        embedded.finite_element(),
        embedded.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<dim, spacedim, LacType>::CopyData copy(
        embedded.finite_element().n_dofs_per_cell());

      for (const auto &cell : embedded.dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto &cell_matrix     = copy.matrices[0];
            auto &cell_rhs        = copy.vectors[0];
            cell_matrix           = 0;
            cell_rhs              = 0;
            const auto &fe_values = scratch.reinit(cell);
            cell->get_dof_indices(copy.local_dof_indices[0]);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  {
                    for (const unsigned int j : fe_values.dof_indices())
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                         fe_values.shape_value(j, q_index) * // phi_j(x_q)
                         fe_values.JxW(q_index));            // dx
                    cell_rhs(i) +=
                      (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                       embedded.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                  }
              }

            embedded.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              embedded.matrix,
              embedded.rhs);
          }

      embedded.matrix.compress(VectorOperation::add);
      embedded.rhs.compress(VectorOperation::add);
      // The rhs of the Lagrange multiplier as a function to plot
      VectorTools::interpolate(embedded.dof_handler,
                               embedded.forcing_term,
                               embedded.solution);
      embedded.solution.compress(VectorOperation::insert);
    }
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(space.timer, "Solve system");

    using BVec       = typename LacType::BlockVector;
    using Vec        = typename BVec::BlockType;
    using LinOp      = LinearOperator<Vec>;
    using BlockLinOp = BlockLinearOperator<BVec>;

    auto A     = linear_operator<Vec>(space.matrix.block(0, 0));
    auto Bt    = linear_operator<Vec>(coupling_matrix);
    auto B     = transpose_operator(Bt);
    auto A_inv = A;
    auto M     = linear_operator<Vec>(embedded.matrix.block(0, 0));
    auto M_inv = M;

    // Extract constant modes for the AMG preconditioner
    std::vector<std::vector<bool>> space_constant_modes;
    if constexpr (std::is_same_v<LacType, LAC::LATrilinos>)
      if (filter_space_averages)
        {
          DoFTools::extract_constant_modes(space.dof_handler,
                                           ComponentMask({true}),
                                           space_constant_modes);
          space.preconditioner.set_constant_modes(space_constant_modes);
        }
    space.preconditioner.initialize(space.matrix.block(0, 0));
    A_inv = space.inverse_operator(A, space.preconditioner);

    embedded.preconditioner.initialize(embedded.matrix.block(0, 0));
    auto M_prec = linear_operator<Vec>(M, embedded.preconditioner);
    M_inv       = mass_solver(M, M_prec);

    auto &lambda       = embedded.solution.block(0);
    auto &embedded_rhs = embedded.rhs.block(0);
    auto &solution     = space.solution.block(0);
    auto &rhs          = space.rhs.block(0);

    if (system_solver_type == "schur")
      {
        auto S      = B * A_inv * Bt;
        auto S_prec = identity_operator(S);
        auto S_inv  = embedded.inverse_operator(S, M_inv);

        lambda   = S_inv * (B * A_inv * rhs - embedded_rhs);
        solution = A_inv * (rhs - Bt * lambda);
      }
    else if (system_solver_type == "diagonal" || system_solver_type == "back" ||
             system_solver_type == "forward")
      {
        auto ZeroM = null_operator(M);

        const auto AA =
          block_operator<2, 2, BVec, BVec>({{{{A, Bt}}, {{B, ZeroM}}}});

        const auto AA_diag_inv = block_diagonal_operator<2, BVec, BVec>(
          std::array<LinOp, 2>{{A_inv, M_inv}});

        BVec temp_rhs, temp_sol;
        AA.reinit_domain_vector(temp_sol, true);
        AA.reinit_range_vector(temp_rhs, true);

        temp_rhs.block(0) = rhs;
        temp_rhs.block(1) = embedded_rhs;
        if (system_solver_type == "diagonal")
          {
            auto AA_inv = embedded.inverse_operator(AA, AA_diag_inv);
            temp_sol    = AA_inv * temp_rhs;
          }
        else if (system_solver_type == "back")
          {
            auto tri_back = block_back_substitution(AA, AA_diag_inv);
            auto AA_inv   = embedded.inverse_operator(AA, tri_back);
            temp_sol      = AA_inv * temp_rhs;
          }
        else if (system_solver_type == "forward")
          {
            auto tri_for = block_forward_substitution(AA, AA_diag_inv);
            auto AA_inv  = embedded.inverse_operator(AA, tri_for);
            temp_sol     = AA_inv * temp_rhs;
          }
        solution = temp_sol.block(0);
        lambda   = temp_sol.block(1);
      }

    // Distribute all constraints.
    embedded.constraints.distribute(lambda);
    embedded.locally_relevant_solution = embedded.solution;
    space.constraints.distribute(solution);
    space.locally_relevant_solution = space.solution;
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::output_results(
    const unsigned int cycle)
  {
    space.output_results(cycle);
    embedded.output_results(cycle);
  }



  template <int spacedim, typename LacType>
  void
  LagrangeMultipliers<spacedim, LacType>::run()
  {
    deallog.depth_console(space.verbosity_level);
    generate_grids();
    for (const auto &cycle : space.grid_refinement.get_refinement_cycles())
      {
        deallog.push("Cycle " + Utilities::int_to_string(cycle));
        setup_system();
        assemble_system();
        solve();
        space.estimate(space.error_per_cell);
        embedded.estimate(embedded.error_per_cell);
        output_results(cycle);
        if (cycle < space.grid_refinement.get_n_refinement_cycles() - 1)
          {
            space.mark(space.error_per_cell);
            space.refine();
            embedded.triangulation.refine_global(1);
            coupling.adjust_grid_refinements(space.triangulation,
                                             embedded.triangulation,
                                             false);
          }
        deallog.pop();
      }
    if (space.mpi_rank == 0)
      {
        space.error_table.output_table(std::cout);
        embedded.error_table.output_table(std::cout);
      }
  }

  // template class LagrangeMultipliers<2>;
  // template class LagrangeMultipliers<3>;

  template class LagrangeMultipliers<2, LAC::LATrilinos>;
  template class LagrangeMultipliers<3, LAC::LATrilinos>;
  template class LagrangeMultipliers<2, LAC::LAPETSc>;
  template class LagrangeMultipliers<3, LAC::LAPETSc>;
} // namespace PDEs
