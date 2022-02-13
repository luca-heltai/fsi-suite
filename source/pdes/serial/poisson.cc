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

      // We propagate the information about the constants to all functions of
      // the problem, so that constants can be used within the functions
      boundary_conditions.update_user_substitution_map(constants);
      exact_solution.update_constants(constants);
      forcing_term.update_constants(constants);

      dof_handler.distribute_dofs(finite_element);

      // Since our code runs both for simplex grids and for hyper-cube grids, we
      // need to make sure that we build the correct mapping for the grid. In
      // this code we actually use a linear mapping, independently on the order
      // of the finite element space.
      mapping = get_default_linear_mapping(triangulation).clone();

      deallog << "Number of dofs " << dof_handler.n_dofs() << std::endl;

      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // We now check that the boundary conditions are consistent with the
      // triangulation object, that is, we check that the boundary indicators
      // specified in the parameter file are actually present in the
      // triangulation, that no boundary indicator is specified twice, and that
      // all boundary ids of the triangulation are actually covered by the
      // parameter file configuration.
      boundary_conditions.check_consistency(triangulation);

      // This is where we apply essential boundary conditions. The
      // ParsedTools::BoundaryConditions class takes care of collecting boundary
      // ids, and calling the appropriate function to apply the boundary
      // condition on the selected boundary ids. Essential boundary conditions
      // need to be incorporated in the constraints of the linear system, since
      // they are part of the definition of the solution space.
      //
      // Natural bondary conditions, on the other hand, need access to the rhs
      // of the problem and are not treated via constraints. We will deal with
      // them later on.
      boundary_conditions.apply_essential_boundary_conditions(*mapping,
                                                              dof_handler,
                                                              constraints);
      constraints.close();

      // Everything here is identical to step-3 in the deal.II tutorials.
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit(sparsity_pattern);
      solution.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());

      // And here ParsedTools::BoundaryConditions kicks in again, and allows you
      // to assemble those boundary condition types that require access to the
      // matrix and to the rhs, after the essential boundary conditions are
      // taken care of.
      boundary_conditions.apply_natural_boundary_conditions(
        *mapping, dof_handler, constraints, system_matrix, system_rhs);
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::assemble_system()
    {
      deallog << "Assemble system" << std::endl;
      const ReferenceCell cell_type = finite_element().reference_cell();

      // Differently from step-3, here we need to change quadrature formula
      // according to the type of cell that the triangulation stores. If it is
      // quads or hexes, then this would be QGauss, otherwise, it will be
      // QGaussSimplex.
      const Quadrature<dim> quadrature_formula =
        cell_type.get_gauss_type_quadrature<dim>(
          finite_element().tensor_degree() + 1);

      // Again, since we support also tria and tets, we need to make sure we
      // pass a compatible mapping to the FEValues object, otherwise things may
      // not work.
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


    // We use a different approach for the solution of the linear system here
    // w.r.t. what is done in step-3. The ParsedLAC::InverseOperator object is
    // used to infer the solver type and the solver parameters from the
    // parameter file, and the same is done for an AMG preconditioner.
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



    // Finally, we output the solution to a file in a format that can be
    // speficied in the parameter file. The structure of this function is
    // identical to step-3, however, here we use the ParsedTools::DataOut class,
    // which specifies the output format, the filename, and many other options
    // from the parameter file, and just add the solution vector using the
    // ParsedTools::DataOut::add_data_vector() function, which works similarly
    // to dealii::DataOut::add_data_vector(), with the exception of the second
    // argument. Here the type of data to produce (i.e., vector, scalar, etc.)
    // is inferred by the repetitions in the component_names arguement, rather
    // than specifying them by hand.
    //
    // This makes the structure of this function almost identical in all
    // programs of the FSI-suite.
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


    // And finally, the run() method. This is the main function of the program.
    // Differently from what is done in step-3, we don't have a separate
    // function for the grid generation, since we let the grid_generator object
    // do the heavy lifting. And differently from what happens in step-3, here
    // we also do several cycles of refinement, in order to compute the
    // convergence rates of the error.
    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::run()
    {
      deallog.pop();
      deallog.depth_console(console_level);
      grid_generator.generate(triangulation);
      for (const auto &cycle : grid_refinement.get_refinement_cycles())
        {
          deallog.push("Cycle " + Utilities::int_to_string(cycle));
          setup_system();
          assemble_system();
          solve();

          // This is is leveraging the ParsedTools::ConvergenceTable class, to
          // compute errors according to what is specified in the parameter file
          error_table.error_from_exact(dof_handler, solution, exact_solution);
          output_results(cycle);

          // Differently from step-3, we also support local refinement. The
          // following function call is only executed if we are not in the last
          // cycle, and implements the actual estimate, mark, refine steps of
          // classical AFEM algorithms, using the parameter file to drive the
          // execution.
          if (cycle < grid_refinement.get_n_refinement_cycles() - 1)
            grid_refinement.estimate_mark_refine(*mapping,
                                                 dof_handler,
                                                 solution,
                                                 triangulation);
          deallog.pop();
        }
      // Make sure we output the error table after the last cycle
      error_table.output_table(std::cout);
    }

    // We explicitly instantiate all of the different combinations of dim and
    // spacedim, so that users can run this in different dimension in the tests
    template class Poisson<1, 1>;
    template class Poisson<1, 2>;
    template class Poisson<1, 3>;
    template class Poisson<2, 2>;
    template class Poisson<2, 3>;
    template class Poisson<3, 3>;
  } // namespace Serial
} // namespace PDEs