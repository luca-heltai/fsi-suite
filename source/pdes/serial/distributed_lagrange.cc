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

#include "pdes/serial/distributed_lagrange.h"

#include <deal.II/base/logstream.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim, typename LacType>
    DistributedLagrange<dim, spacedim, LacType>::DistributedLagrange()
      : ParameterAcceptor("Distributed Lagrange")
      , space("u", "Space")
      , space_cache(space.triangulation)
      , embedded("w", "Embedded")
      , embedded_cache(embedded.triangulation)
      , coupling("/Coupling")
    {
      add_parameter("Use direct solver", use_direct_solver);
    }



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::generate_grids()
    {
      TimerOutput::Scope timer_section(space.timer, "generate_grids_and_fes");
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



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::setup_system()
    {
      space.setup_system();
      embedded.setup_system();

      coupling.assemble_sparsity(coupling_sparsity);
      coupling_matrix.reinit(coupling_sparsity);
    }



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::assemble_system()
    {
      const auto space_quad = ParsedTools::Components::get_cell_quadrature(
        space.triangulation, space.finite_element().tensor_degree() + 1);
      const auto embedded_quad = ParsedTools::Components::get_cell_quadrature(
        embedded.triangulation, embedded.finite_element().tensor_degree() + 1);
      {
        TimerOutput::Scope timer_section(space.timer, "Assemble system");
        // Stiffness matrix and rhs
        MatrixTools::create_laplace_matrix(
          space.dof_handler,
          space_quad,
          space.matrix.block(0, 0),
          space.forcing_term,
          space.rhs.block(0),
          static_cast<const Function<spacedim> *>(nullptr),
          space.constraints);
      }
      {
        TimerOutput::Scope timer_section(space.timer,
                                         "Assemble coupling system");
        coupling.assemble_matrix(coupling_matrix);
      }
      {
        TimerOutput::Scope timer_section(space.timer, "Assemble embedded mass");
        MatrixCreator::create_mass_matrix(
          embedded.dof_handler,
          embedded_quad,
          embedded.matrix,
          static_cast<const Function<spacedim> *>(nullptr),
          embedded.constraints);

        // The rhs of the Lagrange multiplier as a function to plot
        VectorTools::interpolate(embedded.dof_handler,
                                 embedded.forcing_term,
                                 embedded.solution);

        // The rhs of the Lagrange multiplier
        VectorTools::create_right_hand_side(embedded.dof_handler,
                                            embedded_quad,
                                            embedded.forcing_term,
                                            embedded.rhs);
      }
    }



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::solve()
    {
      TimerOutput::Scope timer_section(space.timer, "Solve system");

      auto A     = linear_operator(space.matrix.block(0, 0));
      auto Bt    = linear_operator(coupling_matrix);
      auto B     = transpose_operator(Bt);
      auto A_inv = A;
      auto M     = linear_operator(embedded.matrix.block(0, 0));
      auto M_inv = M;

      typename LacType::DirectSolver A_inv_direct;
      typename LacType::DirectSolver M_inv_direct;

      if (use_direct_solver)
        {
          A_inv_direct.initialize(space.matrix.block(0, 0));
          A_inv = linear_operator(A, A_inv_direct);
          M_inv_direct.initialize(embedded.matrix.block(0, 0));
          M_inv = linear_operator(M, M_inv_direct);
        }
      else
        {
          space.preconditioner.initialize(space.matrix.block(0, 0));
          A_inv = space.inverse_operator(A, space.preconditioner);
        }


      auto &lambda       = embedded.solution.block(0);
      auto &embedded_rhs = embedded.rhs.block(0);
      auto &solution     = space.solution.block(0);
      auto &rhs          = space.rhs.block(0);

      if (false)
        {
          // Solve the system
          solution = A_inv * rhs;
          lambda   = M_inv * B * solution;
        }
      else
        {
          auto S      = B * A_inv * Bt;
          auto S_prec = identity_operator(S);
          auto S_inv  = embedded.inverse_operator(S, M_inv);
          lambda      = S_inv * (B * A_inv * rhs - embedded_rhs);
          solution    = A_inv * (rhs - Bt * lambda);
        }
      embedded.constraints.distribute(lambda);
      embedded.locally_relevant_solution = embedded.solution;
      space.constraints.distribute(solution);
      space.locally_relevant_solution = space.solution;
    }



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::output_results(
      const unsigned int cycle)
    {
      space.output_results(cycle);
      embedded.output_results(cycle);
    }



    template <int dim, int spacedim, typename LacType>
    void
    DistributedLagrange<dim, spacedim, LacType>::run()
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

    template class DistributedLagrange<1, 2>;
    template class DistributedLagrange<2, 2>;
    template class DistributedLagrange<2, 3>;
    template class DistributedLagrange<3, 3>;
  } // namespace Serial

} // namespace PDEs
