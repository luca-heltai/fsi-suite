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

#include "pdes/serial/reduced_lagrange.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "projection_operator.h"

using namespace dealii;

namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim>
    ReducedLagrange<dim, spacedim>::ReducedLagrange()
      : ParameterAcceptor("/")
      , space_dh(space_grid)
      , embedded_dh(embedded_grid)
      , embedded_configuration_dh(embedded_grid)
      , monitor(std::cout,
                TimerOutput::summary,
                TimerOutput::cpu_and_wall_times)
      , grid_generator("/Grid/Ambient")
      , grid_refinement("/Grid/Refinement")
      , embedded_grid_generator("/Grid/Embedded")
      , embedded_mapping(embedded_configuration_dh, "/Grid/Embedded/Mapping")
      , constants("/Functions")
      , embedded_value_function("/Functions", "0", "Embedded value")
      , forcing_term("/Functions", "0", "Forcing term")
      , exact_solution("/Functions", "0", "Exact solution")
      , boundary_conditions("/Boundary conditions")
      , stiffness_inverse_operator("/Solver/Stiffness")
      , stiffness_preconditioner("/Solver/Stiffness AMG")
      , mass_preconditioner("/Solver/Mass AMG")
      , schur_inverse_operator("/Solver/Schur")
      , data_out("/Data out/Space", "output/space")
      , embedded_data_out("/Data out/Embedded", "output/embedded")
    {
      add_parameter("Coupling quadrature order", coupling_quadrature_order);
      add_parameter("Console level", console_level);
      add_parameter("Delta refinement", delta_refinement);
      add_parameter("Number of basis functions", n_basis);
      add_parameter("Finite element degree (ambient space)",
                    finite_element_degree);
      add_parameter("Finite element degree (embedded space)",
                    embedded_space_finite_element_degree);
      add_parameter("Finite element degree (configuration)",
                    embedded_configuration_finite_element_degree);
      enter_subsection("Solver");
      enter_subsection("Stiffness");
      add_parameter("Use direct solver", use_direct_solver);
      leave_subsection();
      leave_subsection();

      enter_subsection("Error table");
      enter_my_subsection(this->prm);
      error_table.add_parameters(this->prm);
      leave_my_subsection(this->prm);
      leave_subsection();
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::generate_grids_and_fes()
    {
      TimerOutput::Scope timer_section(monitor, "generate_grids_and_fes");
      grid_generator.generate(space_grid);
      embedded_grid_generator.generate(embedded_grid);

      space_fe = ParsedTools::Components::get_lagrangian_finite_element(
        space_grid, finite_element_degree);

      embedded_fe = ParsedTools::Components::get_lagrangian_finite_element(
        embedded_grid, embedded_space_finite_element_degree);

      const auto embedded_base_fe =
        ParsedTools::Components::get_lagrangian_finite_element(
          embedded_grid, embedded_space_finite_element_degree);

      embedded_configuration_fe.reset(
        new FESystem<dim, spacedim>(*embedded_base_fe, spacedim));

      space_mapping = std::make_unique<MappingFE<spacedim>>(*space_fe);

      space_grid_tools_cache =
        std::make_unique<GridTools::Cache<spacedim, spacedim>>(space_grid,
                                                               *space_mapping);

      embedded_configuration_dh.distribute_dofs(*embedded_configuration_fe);
      embedded_configuration.reinit(embedded_configuration_dh.n_dofs());
      embedded_mapping.initialize(embedded_configuration);

      adjust_embedded_grid();
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::adjust_embedded_grid(
      const bool apply_delta_refinement)
    {
      namespace bgi = boost::geometry::index;
      // Now get a vector of all cell centers of the embedded grid and refine
      // them  untill every cell is of diameter smaller than the space
      // surrounding cells
      std::vector<
        std::tuple<Point<spacedim>, decltype(embedded_grid.begin_active())>>
        centers;

      for (const auto &cell : embedded_grid.active_cell_iterators())
        centers.emplace_back(
          std::make_tuple(embedded_mapping().get_center(cell), cell));

      // Refine the space grid according to the delta refinement
      if (apply_delta_refinement && delta_refinement != 0)
        for (unsigned int i = 0; i < delta_refinement; ++i)
          {
            const auto &tree =
              space_grid_tools_cache
                ->get_locally_owned_cell_bounding_boxes_rtree();
            for (const auto &[center, cell] : centers)
              for (const auto &[space_box, space_cell] :
                   tree | bgi::adaptors::queried(bgi::contains(center)))
                {
                  space_cell->set_refine_flag();
                  for (const auto face_no : space_cell->face_indices())
                    if (!space_cell->at_boundary(face_no))
                      space_cell->neighbor(face_no)->set_refine_flag();
                }
            space_grid.execute_coarsening_and_refinement();
          }

      bool done = false;
      while (done == false)
        {
          // Now refine the embedded grid if required
          const auto &tree = space_grid_tools_cache
                               ->get_locally_owned_cell_bounding_boxes_rtree();
          // Let's check all cells whose bounding box contains an embedded
          // center
          done          = true;
          namespace bgi = boost::geometry::index;
          for (const auto &[center, cell] : centers)
            {
              const auto &[p1, p2] =
                embedded_mapping().get_bounding_box(cell).get_boundary_points();
              const auto diameter = p1.distance(p2);

              for (const auto &[space_box, space_cell] :
                   tree | bgi::adaptors::queried(bgi::contains(center)))
                if (space_cell->diameter() < diameter)
                  {
                    cell->set_refine_flag();
                    done = false;
                  }
            }
          if (done == false)
            {
              // Compute again the embedded displacement grid
              embedded_grid.execute_coarsening_and_refinement();
              embedded_configuration_dh.distribute_dofs(
                *embedded_configuration_fe);
              embedded_configuration.reinit(embedded_configuration_dh.n_dofs());
              embedded_mapping.initialize(embedded_configuration);

              // Rebuild the centers vector
              centers.clear();

              for (const auto &cell : embedded_grid.active_cell_iterators())
                centers.emplace_back(
                  std::make_tuple(embedded_mapping().get_center(cell), cell));
            }
        }

      const double embedded_space_maximal_diameter =
        GridTools::maximal_cell_diameter(embedded_grid, embedded_mapping());
      const double embedded_space_minimal_diameter =
        GridTools::minimal_cell_diameter(embedded_grid, embedded_mapping());

      double space_minimal_diameter =
        GridTools::minimal_cell_diameter(space_grid);
      double space_maximal_diameter =
        GridTools::maximal_cell_diameter(space_grid);

      deallog << "Space min/max diameters: " << space_minimal_diameter << "/"
              << space_maximal_diameter << std::endl
              << "Embedded space min/max diameters: "
              << embedded_space_minimal_diameter << "/"
              << embedded_space_maximal_diameter << std::endl;
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::update_basis_functions()
    {
      basis_functions.resize(n_basis, Vector<double>(embedded_dh.n_dofs()));

      if (n_basis > 0)
        basis_functions[0] = 1.0;

      for (unsigned int c = 1; c < n_basis; ++c)
        {
          unsigned int k = (c + 1) / 2;
          deallog << "About to build functions." << std::endl;

          FunctionParser<spacedim> b1(
            "sqrt(pi^k*(2*k + 2))*(x^2 + y^2)^(k/2)*cos(k*atan2(y, x))",
            "k=" + std::to_string(k) + ", pi=" + std::to_string(M_PI));
          FunctionParser<spacedim> b2(
            "sqrt(pi^k*(2*k + 2))*(x^2 + y^2)^(k/2)*sin(k*atan2(y, x))",
            "k=" + std::to_string(k) + ", pi=" + std::to_string(M_PI));

          deallog << "Built functions." << std::endl;

          if ((c + 1) % 2 == 0)
            {
              VectorTools::interpolate(embedded_mapping(),
                                       embedded_dh,
                                       b1,
                                       basis_functions[c]);
            }
          else
            {
              VectorTools::interpolate(embedded_mapping(),
                                       embedded_dh,
                                       b2,
                                       basis_functions[c]);
            }
          deallog << "Basis function " << c
                  << " norm: " << basis_functions[c].l2_norm() << std::endl;
        }
    }


    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::setup_dofs()
    {
      TimerOutput::Scope timer_section(monitor, "setup_dofs");

      space_dh.distribute_dofs(*space_fe);
      constraints.clear();
      DoFTools::make_hanging_node_constraints(space_dh, constraints);
      boundary_conditions.apply_essential_boundary_conditions(*space_mapping,
                                                              space_dh,
                                                              constraints);
      constraints.close();
      {
        DynamicSparsityPattern dsp(space_dh.n_dofs(), space_dh.n_dofs());
        DoFTools::make_sparsity_pattern(space_dh, dsp, constraints);
        stiffness_sparsity.copy_from(dsp);
        stiffness_matrix.reinit(stiffness_sparsity);
      }

      solution.reinit(space_dh.n_dofs());
      rhs.reinit(space_dh.n_dofs());
      deallog << "Embedding dofs: " << space_dh.n_dofs() << std::endl;


      embedded_dh.distribute_dofs(*embedded_fe);
      embedded_constraints.clear();
      DoFTools::make_hanging_node_constraints(embedded_dh,
                                              embedded_constraints);
      embedded_constraints.close();
      {
        DynamicSparsityPattern dsp(embedded_dh.n_dofs(), embedded_dh.n_dofs());
        DoFTools::make_sparsity_pattern(embedded_dh, dsp, embedded_constraints);
        embedded_sparsity.copy_from(dsp);
        embedded_mass_matrix.reinit(embedded_sparsity);
      }

      lambda.reinit(embedded_dh.n_dofs());
      embedded_rhs.reinit(embedded_dh.n_dofs());
      embedded_value.reinit(embedded_dh.n_dofs());
      deallog << "Embedded dofs: " << embedded_dh.n_dofs() << std::endl;
      deallog << "Reduced dofs: " << n_basis << std::endl;

      reduced_rhs.reinit(n_basis);
      reduced_value.reinit(n_basis);
      reduced_lambda.reinit(n_basis);

      boundary_conditions.apply_natural_boundary_conditions(
        *space_mapping, space_dh, constraints, stiffness_matrix, rhs);
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::setup_coupling()
    {
      TimerOutput::Scope timer_section(monitor, "Setup coupling");
      const auto embedded_quad = ParsedTools::Components::get_cell_quadrature(
        embedded_grid, embedded_fe->tensor_degree() + 1);

      DynamicSparsityPattern dsp(space_dh.n_dofs(), embedded_dh.n_dofs());
      NonMatching::create_coupling_sparsity_pattern(*space_grid_tools_cache,
                                                    space_dh,
                                                    embedded_dh,
                                                    embedded_quad,
                                                    dsp,
                                                    constraints,
                                                    ComponentMask(),
                                                    ComponentMask(),
                                                    embedded_mapping());
      coupling_sparsity.copy_from(dsp);
      coupling_matrix.reinit(coupling_sparsity);
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::assemble_system()
    {
      const auto space_quad = ParsedTools::Components::get_cell_quadrature(
        space_grid, space_fe->tensor_degree() + 1);
      const auto embedded_quad = ParsedTools::Components::get_cell_quadrature(
        embedded_grid, embedded_fe->tensor_degree() + 1);
      {
        TimerOutput::Scope timer_section(monitor, "Assemble system");
        // Stiffness matrix and rhs
        MatrixTools::create_laplace_matrix(
          space_dh,
          space_quad,
          stiffness_matrix,
          forcing_term,
          rhs,
          static_cast<const Function<spacedim> *>(nullptr),
          constraints);

        // Mass matrix
        MatrixCreator::create_mass_matrix(
          embedded_mapping(),
          embedded_dh,
          embedded_quad,
          embedded_mass_matrix,
          static_cast<const Function<spacedim> *>(nullptr),
          embedded_constraints);
      }
      {
        TimerOutput::Scope timer_section(monitor, "Assemble coupling system");
        NonMatching::create_coupling_mass_matrix(*space_grid_tools_cache,
                                                 space_dh,
                                                 embedded_dh,
                                                 embedded_quad,
                                                 coupling_matrix,
                                                 constraints,
                                                 ComponentMask(),
                                                 ComponentMask(),
                                                 embedded_mapping());

        // The rhs of the Lagrange multiplier as a function to plot
        VectorTools::interpolate(embedded_mapping(),
                                 embedded_dh,
                                 embedded_value_function,
                                 embedded_value);

        // The rhs of the Lagrange multiplier
        VectorTools::create_right_hand_side(embedded_mapping(),
                                            embedded_dh,
                                            embedded_quad,
                                            embedded_value_function,
                                            embedded_rhs);
      }
      update_basis_functions();
    }
    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(monitor, "Solve system");

      auto                A     = linear_operator(stiffness_matrix);
      auto                Bt    = linear_operator(coupling_matrix);
      auto                B     = transpose_operator(Bt);
      auto                A_inv = A;
      SparseDirectUMFPACK A_inv_direct;

      if (use_direct_solver)
        {
          A_inv_direct.initialize(stiffness_matrix);
          A_inv = linear_operator(A, A_inv_direct);
        }
      else
        {
          stiffness_preconditioner.initialize(stiffness_matrix);
          A_inv = stiffness_inverse_operator(A, stiffness_preconditioner);
        }

      auto M = linear_operator(embedded_mass_matrix);
      mass_preconditioner.initialize(embedded_mass_matrix);
      auto Minv = linear_operator(M, mass_preconditioner);

      auto S      = B * A_inv * Bt;
      auto S_prec = identity_operator(S);
      // S_prec      = B * Minv * A * Minv * Bt;
      auto S_inv = schur_inverse_operator(S, S_prec);

      if (n_basis == 0)
        {
          lambda   = S_inv * (B * A_inv * rhs - embedded_rhs);
          solution = A_inv * (rhs - Bt * lambda);
        }
      else
        {
          Vector<double> reduced(n_basis);
          std::vector<std::reference_wrapper<const Vector<double>>> basis(
            basis_functions.begin(), basis_functions.end());

          FullMatrix<double> G(n_basis, n_basis);
          for (unsigned int i = 0; i < n_basis; ++i)
            {
              embedded_value = M * basis_functions[i];
              for (unsigned int j = 0; j < n_basis; ++j)
                G(i, j) = basis_functions[j] * embedded_value;
            }
          FullMatrix<double> Ginv(n_basis, n_basis);
          Ginv.invert(G);

          auto R  = projection_operator(reduced, basis);
          auto Rt = transpose_operator(R);

          auto Ct      = Bt * Rt;
          auto C       = R * B;
          auto RS      = C * A_inv * Ct;
          auto RS_prec = identity_operator(RS);
          RS_prec      = C * A * Ct;
          auto RS_inv  = schur_inverse_operator(RS, RS_prec);
          reduced_rhs  = R * embedded_rhs;
          Ginv.vmult(reduced_value, reduced_rhs);
          reduced_lambda = RS_inv * (C * A_inv * rhs - reduced_rhs);
          solution       = A_inv * (rhs - Ct * reduced_lambda);
          embedded_value = Rt * reduced_value;
          lambda         = Rt * reduced_lambda;

          deallog << "Reduced lambda: " << reduced_lambda << std::endl;
          deallog << "Reduced value: " << reduced_value << std::endl;
          deallog << "Reduced rhs: " << reduced_rhs << std::endl;
        }
      constraints.distribute(solution);
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::output_results(const unsigned int cycle)
    {
      TimerOutput::Scope timer_section(monitor, "Output results");
      const auto         suffix =
        Utilities::int_to_string(cycle,
                                 Utilities::needed_digits(
                                   grid_refinement.get_n_refinement_cycles()));

      data_out.attach_dof_handler(space_dh, suffix);
      data_out.add_data_vector(solution, component_names);
      data_out.write_data_and_clear(*space_mapping);

      embedded_data_out.attach_dof_handler(embedded_dh, suffix);
      embedded_data_out.add_data_vector(
        lambda, "lambda", dealii::DataOut<dim, spacedim>::type_dof_data);
      embedded_data_out.add_data_vector(
        embedded_value, "g", dealii::DataOut<dim, spacedim>::type_dof_data);

      unsigned int c = 0;
      for (const auto &basis : basis_functions)
        {
          embedded_data_out.add_data_vector(
            basis,
            "phi_" + std::to_string(c++),
            dealii::DataOut<dim, spacedim>::type_dof_data);
        }
      embedded_data_out.write_data_and_clear(embedded_mapping);
    }



    template <int dim, int spacedim>
    void
    ReducedLagrange<dim, spacedim>::run()
    {
      deallog.depth_console(console_level);
      generate_grids_and_fes();
      for (const auto &cycle : grid_refinement.get_refinement_cycles())
        {
          deallog.push("Cycle " + Utilities::int_to_string(cycle));
          setup_dofs();
          setup_coupling();
          assemble_system();
          solve();
          error_table.error_from_exact(space_dh, solution, exact_solution);
          output_results(cycle);
          if (cycle < grid_refinement.get_n_refinement_cycles() - 1)
            {
              grid_refinement.estimate_mark_refine(space_dh,
                                                   solution,
                                                   space_grid);
              adjust_embedded_grid(false);
            }
          deallog.pop();
        }
      error_table.output_table(std::cout);
    }

    template class ReducedLagrange<1, 2>;
    template class ReducedLagrange<2, 2>;
    template class ReducedLagrange<2, 3>;
    template class ReducedLagrange<3, 3>;
  } // namespace Serial
} // namespace PDEs
