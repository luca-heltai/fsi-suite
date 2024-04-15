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

#include "pdes/coupled_problem.h"

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
  template <int dim, int dim1, int spacedim>
  CoupledProblem<dim, dim1, spacedim>::CoupledProblem(
    const std::string &coupled_problem_name,
    const std::string &primary_problem_variables,
    const std::string &primary_problem_name,
    const std::string &secondary_problem_variables,
    const std::string &secondary_problem_name)
    : ParameterAcceptor(coupled_problem_name)
    , primary_problem(primary_problem_variables, primary_problem_name)
    , primary_problem_cache(primary_problem.triangulation)
    , secondary_problem(secondary_problem_variables, secondary_problem_name)
    , secondary_problem_cache(secondary_problem.triangulation)
  {
    static_assert(dim1 <= dim, "dim1 must be less than or equal to dim");
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::generate_grids()
  {
    TimerOutput::Scope timer_section(primary_problem.timer, "generate_grids");
    primary_problem.grid_generator.generate(primary_problem.triangulation);
    secondary_problem.grid_generator.generate(secondary_problem.triangulation);
    // coupling.initialize(primary_problem_cache,
    //                     primary_problem.dof_handler,
    //                     primary_problem.constraints,
    //                     secondary_problem_cache,
    //                     secondary_problem.dof_handler,
    //                     secondary_problem.constraints);
    // coupling.adjust_grid_refinements(primary_problem.triangulation,
    //                                  secondary_problem.triangulation,
    //                                  true);
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::setup_sparsity(
    BlockDynamicSparsityPattern &dsp)
  {
    (void)dsp;
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::setup_system()
  {
    primary_problem.setup_system();
    secondary_problem.setup_system();

    // Join the two constraints
    CouplingUtilities::join_constraints(primary_problem.constraints,
                                        secondary_problem.constraints,
                                        constraints);

    // Dofs per block of the global problem
    dofs_per_block = primary_problem.dofs_per_block;
    dofs_per_block.insert(dofs_per_block.end(),
                          secondary_problem.dofs_per_block.begin(),
                          secondary_problem.dofs_per_block.end());

    // Locally owned dofs per block of the global problem
    locally_owned_dofs_per_block = primary_problem.locally_owned_dofs;
    locally_owned_dofs_per_block.insert(
      locally_owned_dofs_per_block.end(),
      secondary_problem.locally_owned_dofs.begin(),
      secondary_problem.locally_owned_dofs.end());

    // Locally relevant dofs per block of the global problem
    locally_relevant_dofs_per_block = primary_problem.locally_relevant_dofs;
    locally_relevant_dofs_per_block.insert(
      locally_relevant_dofs_per_block.end(),
      secondary_problem.locally_relevant_dofs.begin(),
      secondary_problem.locally_relevant_dofs.end());

    LAC::BlockInitializer       init(dofs_per_block,
                               locally_owned_dofs_per_block,
                               locally_relevant_dofs_per_block,
                               primary_problem.mpi_communicator);
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    setup_sparsity(dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               primary_problem.mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs_per_block,
                         dsp,
                         primary_problem.mpi_communicator);
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::assemble_system()
  {
    {
      TimerOutput::Scope timer_section(primary_problem.timer,
                                       "Assemble stiffness system");
      // Stiffness matrix and rhs
      typename LinearProblem<dim, spacedim>::ScratchData scratch(
        *primary_problem.mapping,
        primary_problem.finite_element(),
        primary_problem.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<dim, spacedim>::CopyData copy(
        primary_problem.finite_element().n_dofs_per_cell());

      for (const auto &cell :
           primary_problem.dof_handler.active_cell_iterators())
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
                       primary_problem.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                  }
              }

            primary_problem.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              primary_problem.matrix,
              primary_problem.rhs);
          }

      primary_problem.matrix.compress(VectorOperation::add);
      primary_problem.rhs.compress(VectorOperation::add);
    }
    {
      TimerOutput::Scope timer_section(primary_problem.timer,
                                       "Assemble coupling system");
      // coupling_matrix = 0.0;
      // coupling.assemble_matrix(coupling_matrix);
      // coupling_matrix.compress(VectorOperation::add);
    }
    {
      // Embedded mass matrix and rhs
      typename LinearProblem<dim1, spacedim>::ScratchData scratch(
        *secondary_problem.mapping,
        secondary_problem.finite_element(),
        secondary_problem.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<dim1, spacedim>::CopyData copy(
        secondary_problem.finite_element().n_dofs_per_cell());

      for (const auto &cell :
           secondary_problem.dof_handler.active_cell_iterators())
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
                       secondary_problem.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                  }
              }

            secondary_problem.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              secondary_problem.matrix,
              secondary_problem.rhs);
          }

      secondary_problem.matrix.compress(VectorOperation::add);
      secondary_problem.rhs.compress(VectorOperation::add);
      // The rhs of the Lagrange multiplier as a function to plot
      VectorTools::interpolate(secondary_problem.dof_handler,
                               secondary_problem.forcing_term,
                               secondary_problem.solution);
      secondary_problem.solution.compress(VectorOperation::insert);
    }
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::solve()
  {
    TimerOutput::Scope timer_section(primary_problem.timer, "Solve system");

    // using BlockVectorType       = typename LacType::BlockVector;
    // using VectorType        = typename BlockVectorType::BlockType;
    using LinOp      = LinearOperator<VectorType>;
    using BlockLinOp = BlockLinearOperator<BlockVectorType>;

    auto A = linear_operator<VectorType>(primary_problem.matrix.block(0, 0));
    // auto Bt = linear_operator<VectorType>(coupling_matrix);
    auto Bt    = A;
    auto B     = transpose_operator(Bt);
    auto A_inv = A;
    auto M = linear_operator<VectorType>(secondary_problem.matrix.block(0, 0));
    auto M_inv = M;

    primary_problem.preconditioner.initialize(
      primary_problem.matrix.block(0, 0));
    A_inv = primary_problem.inverse_operator(A, primary_problem.preconditioner);

    secondary_problem.preconditioner.initialize(
      secondary_problem.matrix.block(0, 0));
    auto M_prec =
      linear_operator<VectorType>(M, secondary_problem.preconditioner);
    // M_inv = mass_solver(M, M_prec);

    auto &lambda       = secondary_problem.solution.block(0);
    auto &embedded_rhs = secondary_problem.rhs.block(0);
    auto &solution     = primary_problem.solution.block(0);
    auto &rhs          = primary_problem.rhs.block(0);

    auto S      = B * A_inv * Bt;
    auto S_prec = identity_operator(S);
    auto S_inv  = secondary_problem.inverse_operator(S, M_inv);

    lambda   = S_inv * (B * A_inv * rhs - embedded_rhs);
    solution = A_inv * (rhs - Bt * lambda);

    // Distribute all constraints.
    secondary_problem.constraints.distribute(lambda);
    secondary_problem.locally_relevant_solution = secondary_problem.solution;
    primary_problem.constraints.distribute(solution);
    primary_problem.locally_relevant_solution = primary_problem.solution;
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::output_results(const unsigned int cycle)
  {
    primary_problem.output_results(cycle);
    secondary_problem.output_results(cycle);
  }



  template <int dim, int dim1, int spacedim>
  void
  CoupledProblem<dim, dim1, spacedim>::run()
  {
    deallog.depth_console(primary_problem.verbosity_level);
    generate_grids();
    for (const auto &cycle :
         primary_problem.grid_refinement.get_refinement_cycles())
      {
        deallog.push("Cycle " + Utilities::int_to_string(cycle));
        setup_system();
        assemble_system();
        solve();
        primary_problem.estimate(primary_problem.error_per_cell);
        secondary_problem.estimate(secondary_problem.error_per_cell);
        output_results(cycle);
        if (cycle <
            primary_problem.grid_refinement.get_n_refinement_cycles() - 1)
          {
            primary_problem.mark(primary_problem.error_per_cell);
            primary_problem.refine();
            secondary_problem.triangulation.refine_global(1);
            // coupling.adjust_grid_refinements(primary_problem.triangulation,
            //                                  secondary_problem.triangulation,
            //                                  false);
          }
        deallog.pop();
      }
    if (primary_problem.mpi_rank == 0)
      {
        primary_problem.error_table.output_table(std::cout);
        secondary_problem.error_table.output_table(std::cout);
      }
  }

  template class CoupledProblem<1, 1, 2>;
  template class CoupledProblem<2, 1, 2>;
  template class CoupledProblem<2, 2, 2>;

  template class CoupledProblem<1, 1, 3>;
  template class CoupledProblem<2, 1, 3>;
  template class CoupledProblem<3, 1, 3>;
  template class CoupledProblem<2, 2, 3>;
  template class CoupledProblem<3, 2, 3>;

  template class CoupledProblem<3, 3, 3>;


} // namespace PDEs
