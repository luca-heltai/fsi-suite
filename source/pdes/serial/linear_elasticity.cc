#include "pdes/serial/linear_elasticity.h"

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/error_estimator.h>


using namespace dealii;
namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim>
    LinearElasticity<dim, spacedim>::LinearElasticity()
      : ParameterAcceptor("/LinearElasticity/")
      , component_names(
          ParsedTools::Components::blocks_to_names({"u"}, {spacedim}))
      , grid_generator("/LinearElasticity/Grid")
      , grid_refinement("/LinearElasticity/Grid/Refinement")
      , finite_element("/LinearElasticity/",
                       component_names,
                       "FESystem[FE_Q(1)^d]")
      , dof_handler(triangulation)
      , inverse_operator("/LinearElasticity/Solver")
      , preconditioner("/LinearElasticity/Solver/AMG Preconditioner")
      , constants("/LinearElasticity/Constants",
                  {"lambda", "mu"},
                  {1.0, 1.0},
                  {"Lame parameter", "Lame parameter"})
      , forcing_term(
          "/LinearElasticity/Functions",
          ParsedTools::Components::join(std::vector<std::string>(spacedim, "0"),
                                        ";"),
          "Forcing term")
      , exact_solution(
          "/LinearElasticity/Functions",
          ParsedTools::Components::join(std::vector<std::string>(spacedim, "0"),
                                        ";"),
          "Exact solution")
      , boundary_conditions("/LinearElasticity/Boundary conditions",
                            component_names)
      , error_table(std::vector<std::string>(spacedim, "u"),
                    {{VectorTools::H1_norm, VectorTools::L2_norm}})
      , data_out("/LinearElasticity/Output")
      , displacement(0)
    {
      enter_subsection("Error table");
      enter_my_subsection(this->prm);
      error_table.add_parameters(this->prm);
      leave_my_subsection(this->prm);
      leave_subsection();
      add_parameter("Console level", this->console_level);
    }



    template <int dim, int spacedim>
    void
    LinearElasticity<dim, spacedim>::setup_system()
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
    LinearElasticity<dim, spacedim>::assemble_system()
    {
      deallog << "Assemble system" << std::endl;
      const ReferenceCell cell_type = finite_element().reference_cell();

      const Quadrature<dim> quadrature_formula =
        cell_type.get_gauss_type_quadrature<dim>(
          finite_element().tensor_degree() + 1);

      Quadrature<dim - 1> face_quadrature_formula;
      if constexpr (dim > 1)
        {
          // TODO: make sure we work also for wedges and pyramids
          const ReferenceCell face_type = cell_type.face_reference_cell(0);
          face_quadrature_formula =
            face_type.get_gauss_type_quadrature<dim - 1>(
              finite_element().tensor_degree() + 1);
        }
      else
        {
          face_quadrature_formula =
            QGauss<dim - 1>(finite_element().tensor_degree() + 1);
        }

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
                {
                  const auto &eps_v =
                    fe_values[displacement].symmetric_gradient(i, q_index);
                  const auto &div_v =
                    fe_values[displacement].divergence(i, q_index);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      const auto &eps_u =
                        fe_values[displacement].symmetric_gradient(j, q_index);
                      const auto &div_u =
                        fe_values[displacement].divergence(j, q_index);

                      cell_matrix(i, j) +=
                        (constants["mu"] * eps_v * eps_u +
                         constants["lambda"] * div_v * div_u) *
                        fe_values.JxW(q_index); // dx
                    }

                  cell_rhs(i) +=
                    (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                     forcing_term.value(fe_values.quadrature_point(q_index),
                                        finite_element()
                                          .system_to_component_index(i)
                                          .first) * // f(x_q)
                     fe_values.JxW(q_index));       // dx
                }
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
    LinearElasticity<dim, spacedim>::solve()
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
    LinearElasticity<dim, spacedim>::output_results(const unsigned cycle) const
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
    LinearElasticity<dim, spacedim>::run()
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

    template class LinearElasticity<1, 1>;
    template class LinearElasticity<1, 2>;
    template class LinearElasticity<1, 3>;
    template class LinearElasticity<2, 2>;
    template class LinearElasticity<2, 3>;
    template class LinearElasticity<3, 3>;
  } // namespace Serial
} // namespace PDEs