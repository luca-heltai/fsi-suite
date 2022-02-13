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

#include "pdes/serial/poisson.h"

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/error_estimator.h>


using namespace dealii;
namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim>
    Poisson<dim, spacedim>::Poisson()
      : ParameterAcceptor("/Poisson/")
      , grid_generator("/Poisson/Grid")
      , grid_refinement("/Poisson/Grid/Refinement")
      , finite_element("/Poisson/", "u", "FE_Q(1)")
      , dof_handler(triangulation)
      , inverse_operator("/Poisson/Solver")
      , preconditioner("/Poisson/Solver/AMG Preconditioner")
      , constants("/Poisson/Constants",
                  {"kappa"},
                  {1.0},
                  {"Diffusion coefficient"})
      , forcing_term("/Poisson/Functions",
                     "kappa*8*PI^2*sin(2*PI*x)*sin(2*PI*y)",
                     "Forcing term")
      , exact_solution("/Poisson/Functions",
                       "sin(2*PI*x)*sin(2*PI*y)",
                       "Exact solution")
      , boundary_conditions("/Poisson/Boundary conditions")
      , error_table("/Poisson/Error table")
      , data_out("/Poisson/Output")
    {
      add_parameter("Console level", this->console_level);
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::setup_system()
    {
      deallog << "System setup" << std::endl;
      // No mixed grids
      const auto ref_cells = triangulation.get_reference_cells();
      AssertThrow(
        ref_cells.size() == 1,
        ExcMessage(
          "This program does nots support mixed simplx/hex grid types."));

      // Compatible FE space and grid.
      AssertThrow(finite_element().reference_cell() == ref_cells[0],
                  ExcMessage("The finite element must be defined on the same "
                             "cell type as the grid."));

      boundary_conditions.update_user_substitution_map(constants);
      exact_solution.update_constants(constants);
      forcing_term.update_constants(constants);

      dof_handler.distribute_dofs(finite_element);
      mapping = get_default_linear_mapping(triangulation).clone();

      deallog << "Number of dofs " << dof_handler.n_dofs() << std::endl;

      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      boundary_conditions.check_consistency(triangulation);
      boundary_conditions.apply_essential_boundary_conditions(*mapping,
                                                              dof_handler,
                                                              constraints);
      constraints.close();


      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit(sparsity_pattern);
      solution.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());
      boundary_conditions.apply_natural_boundary_conditions(
        *mapping, dof_handler, constraints, system_matrix, system_rhs);
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::assemble_system()
    {
      deallog << "Assemble system" << std::endl;
      const ReferenceCell cell_type = finite_element().reference_cell();

      const Quadrature<dim> quadrature_formula =
        cell_type.get_gauss_type_quadrature<dim>(
          finite_element().tensor_degree() + 1);

      FEValues<dim, spacedim> fe_values(*mapping,
                                        finite_element,
                                        quadrature_formula,
                                        update_values | update_gradients |
                                          update_quadrature_points |
                                          update_JxW_values);

      const unsigned int dofs_per_cell = finite_element().n_dofs_per_cell();
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (constants["kappa"] *
                     fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx
              for (const unsigned int i : fe_values.dof_indices())
                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   forcing_term.value(
                     fe_values.quadrature_point(q_index)) * // f(x_q)
                   fe_values.JxW(q_index));                 // dx
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::solve()
    {
      deallog << "Solve system" << std::endl;
      preconditioner.initialize(system_matrix);
      const auto A    = linear_operator<Vector<double>>(system_matrix);
      const auto Ainv = inverse_operator(A, preconditioner);
      solution        = Ainv * system_rhs;
      constraints.distribute(solution);
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::output_results(const unsigned cycle) const
    {
      deallog << "Output results" << std::endl;
      // Save each cycle in its own file
      const auto suffix =
        Utilities::int_to_string(cycle,
                                 Utilities::needed_digits(
                                   grid_refinement.get_n_refinement_cycles()));
      data_out.attach_dof_handler(dof_handler, suffix);
      data_out.add_data_vector(solution, component_names);
      data_out.write_data_and_clear(*mapping);
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::run()
    {
      deallog.pop();
      deallog.depth_console(console_level);
      grid_generator.generate(triangulation);
      for (unsigned int cycle = 0;
           cycle < grid_refinement.get_n_refinement_cycles();
           ++cycle)
        {
          deallog.push("Cycle " + Utilities::int_to_string(cycle));
          setup_system();
          assemble_system();
          solve();
          error_table.error_from_exact(dof_handler, solution, exact_solution);
          output_results(cycle);
          if (cycle < grid_refinement.get_n_refinement_cycles() - 1)
            grid_refinement.estimate_mark_refine(*mapping,
                                                 dof_handler,
                                                 solution,
                                                 triangulation);
          deallog.pop();
        }
      error_table.output_table(std::cout);
    }

    template class Poisson<1, 1>;
    template class Poisson<1, 2>;
    template class Poisson<1, 3>;
    template class Poisson<2, 2>;
    template class Poisson<2, 3>;
    template class Poisson<3, 3>;
  } // namespace Serial
} // namespace PDEs