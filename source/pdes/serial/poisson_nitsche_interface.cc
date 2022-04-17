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

#include "pdes/serial/poisson_nitsche_interface.h"

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/error_estimator.h>


using namespace dealii;
namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim>
    PoissonNitscheInterface<dim, spacedim>::PoissonNitscheInterface()
      : ParameterAcceptor("/PoissonNitscheInterface/")
      , grid_generator("/PoissonNitscheInterface/Grid/Ambient")
      , embedded_grid_generator("/PoissonNitscheInterface/Grid/Embedded")
      , grid_refinement("/PoissonNitscheInterface/Grid/Refinement")
      , space_fe("/PoissonNitscheInterface/", "u", "FE_Q(1)")
      , space_dh(space_triangulation)
      , inverse_operator("/PoissonNitscheInterface/Solver")
      , preconditioner("/PoissonNitscheInterface/Solver/AMG Preconditioner")
      , constants("/PoissonNitscheInterface/Constants",
                  {"kappa"},
                  {1.0},
                  {"Diffusion coefficient"})
      , forcing_term("/PoissonNitscheInterface/Functions",
                     "kappa*8*PI^2*sin(2*PI*x)*sin(2*PI*y)",
                     "Forcing term")
      , embedded_value("/PoissonNitscheInterface/Functions",
                       "1.0",
                       "Embedded value")
      , nitsche_coefficient("/PoissonNitscheInterface/Functions",
                            "2.0",
                            "Nitsche cofficient")
      , exact_solution("/PoissonNitscheInterface/Functions",
                       "sin(2*PI*x)*sin(2*PI*y)",
                       "Exact solution")
      , boundary_conditions("/PoissonNitscheInterface/Boundary conditions")
      , timer(deallog.get_console(),
              TimerOutput::summary,
              TimerOutput::cpu_and_wall_times)
      , error_table("/PoissonNitscheInterface/Error table")
      , data_out("/PoissonNitscheInterface/Output")
    {
      add_parameter("Console level", this->console_level);
    }



    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::generate_grids()
    {
      TimerOutput::Scope timer_section(timer, "Generate grids");
      grid_generator.generate(space_triangulation);
      embedded_grid_generator.generate(embedded_triangulation);
      // We create unique pointers to cached triangulations. This This objects
      // will be necessary to compute the the Quadrature formulas on the
      // intersection of the cells.
      space_cache = std::make_unique<GridTools::Cache<spacedim, spacedim>>(
        space_triangulation);
      embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
        embedded_triangulation);
    }



    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::setup_system()
    {
      TimerOutput::Scope timer_section(timer, "Setup system");
      deallog << "System setup" << std::endl;
      // No mixed grids
      const auto ref_cells = space_triangulation.get_reference_cells();
      AssertThrow(
        ref_cells.size() == 1,
        ExcMessage(
          "This program does nots support mixed simplx/hex grid types."));

      // Compatible FE space and grid.
      AssertThrow(space_fe().reference_cell() == ref_cells[0],
                  ExcMessage("The finite element must be defined on the same "
                             "cell type as the grid."));

      // We propagate the information about the constants to all functions of
      // the problem, so that constants can be used within the functions
      boundary_conditions.update_user_substitution_map(constants);
      exact_solution.update_constants(constants);
      forcing_term.update_constants(constants);
      embedded_value.update_constants(constants);
      space_dh.distribute_dofs(space_fe);


      mapping = get_default_linear_mapping(space_triangulation).clone();

      deallog << "Number of dofs " << space_dh.n_dofs() << std::endl;

      space_constraints.clear();
      DoFTools::make_hanging_node_constraints(space_dh, space_constraints);


      boundary_conditions.check_consistency(space_triangulation);

      // This is where we apply essential boundary conditions.
      boundary_conditions.apply_essential_boundary_conditions(
        *mapping, space_dh, space_constraints);
      space_constraints.close();


      DynamicSparsityPattern dsp(space_dh.n_dofs());
      DoFTools::make_sparsity_pattern(space_dh, dsp, space_constraints, false);
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit(sparsity_pattern);
      solution.reinit(space_dh.n_dofs());
      system_rhs.reinit(space_dh.n_dofs());
    }



    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::assemble_system()
    {
      {
        TimerOutput::Scope timer_section(timer, "Assemble system");
        deallog << "Assemble system" << std::endl;
        const ReferenceCell cell_type = space_fe().reference_cell();


        const Quadrature<spacedim> quadrature_formula =
          cell_type.get_gauss_type_quadrature<spacedim>(
            space_fe().tensor_degree() + 1);


        FEValues<spacedim, spacedim> fe_values(*mapping,
                                               space_fe,
                                               quadrature_formula,
                                               update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values);

        const unsigned int dofs_per_cell = space_fe().n_dofs_per_cell();
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        for (const auto &cell : space_dh.active_cell_iterators())
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
            space_constraints.distribute_local_to_global(cell_matrix,
                                                         cell_rhs,
                                                         local_dof_indices,
                                                         system_matrix,
                                                         system_rhs);
          }
      }


      deallog << "Assemble Nitsche contributions" << '\n';
      {
        TimerOutput::Scope timer_section(timer, "Assemble Nitsche terms");

        // Add the Nitsche's contribution to the system matrix. The coefficient
        // that multiplies the inner product is equal to 2.0, and the penalty is
        // set to 100.0.
        NonMatching::
          assemble_nitsche_with_exact_intersections<spacedim, dim, spacedim>(
            space_dh,
            cells_and_quads,
            system_matrix,
            space_constraints,
            ComponentMask(),
            MappingQ1<spacedim, spacedim>(),
            nitsche_coefficient,
            penalty);

        // Add the Nitsche's contribution to the rhs. The embedded value is
        // parsed from the parameter file, while we have again the constant 2.0
        // in front of that term, parsed as above from command line. Finally, we
        // have the penalty parameter as before.
        NonMatching::
          create_nitsche_rhs_with_exact_intersections<spacedim, dim, spacedim>(
            space_dh,
            cells_and_quads,
            system_rhs,
            space_constraints,
            MappingQ1<spacedim, spacedim>(),
            embedded_value,
            ConstantFunction<spacedim>(2.0),
            penalty);
      }
    }


    // We solve the resulting system as done in the classical Poisson example.
    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(timer, "Solve system");
      deallog << "Solve system" << std::endl;
      preconditioner.initialize(system_matrix);
      const auto A    = linear_operator<Vector<double>>(system_matrix);
      const auto Ainv = inverse_operator(A, preconditioner);
      solution        = Ainv * system_rhs;
      space_constraints.distribute(solution);
    }



    // Finally, we output the solution living in the embedding space, just
    // like all the other programs.
    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::output_results(
      const unsigned cycle) const
    {
      TimerOutput::Scope timer_section(timer, "Output results");
      deallog << "Output results" << std::endl;
      // Save each cycle in its own file
      const auto suffix =
        Utilities::int_to_string(cycle,
                                 Utilities::needed_digits(
                                   grid_refinement.get_n_refinement_cycles()));
      data_out.attach_dof_handler(space_dh, suffix);
      data_out.add_data_vector(solution, component_names);
      data_out.write_data_and_clear(*mapping);
      // Save the grids
      {
        std::ofstream output_test_space("space_grid.vtk");
        GridOut().write_vtk(space_triangulation, output_test_space);
        std::ofstream output_test_embedded("embedded_grid.vtk");
        GridOut().write_vtk(embedded_triangulation, output_test_embedded);
      }
    }


    // The run() method here differs only in the call to
    // NonMatching::compute_intersection().
    template <int dim, int spacedim>
    void
    PoissonNitscheInterface<dim, spacedim>::run()
    {
      deallog.pop();
      deallog.depth_console(console_level);

      generate_grids();
      for (const auto &cycle : grid_refinement.get_refinement_cycles())
        {
          deallog.push("Cycle " + Utilities::int_to_string(cycle));



          // Here we compute all the things we need to assemble the Nitsche's
          // contributions, namely the two cached triangulations and a degree to
          // integrate over the intersections.
          cells_and_quads = NonMatching::compute_intersection(
            *space_cache, *embedded_cache, 2 * space_fe().tensor_degree() + 1);
          setup_system();
          assemble_system();
          solve();


          error_table.error_from_exact(space_dh, solution, exact_solution);
          output_results(cycle);


          if (cycle < grid_refinement.get_n_refinement_cycles() - 1)
            grid_refinement.estimate_mark_refine(*mapping,
                                                 space_dh,
                                                 solution,
                                                 space_triangulation);


          deallog.pop();
        }
      // Make sure we output the error table after the last cycle
      error_table.output_table(std::cout);
    }

    // We explicitly instantiate all of the different combinations of dim and
    // spacedim, so that users can run this in different dimension in the tests
    template class PoissonNitscheInterface<1, 2>;
    template class PoissonNitscheInterface<2, 2>;
    template class PoissonNitscheInterface<2, 3>;
    template class PoissonNitscheInterface<3, 3>;

  } // namespace Serial
} // namespace PDEs
