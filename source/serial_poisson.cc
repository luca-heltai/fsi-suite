#include "serial_poisson.h"

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/error_estimator.h>


using namespace dealii;
namespace PDEs
{
  template <int dim, int spacedim>
  SerialPoisson<dim, spacedim>::SerialPoisson()
    : ParameterAcceptor("/Serial Poisson/")
    , grid_generator("/Serial Poisson/Grid")
    , grid_refinement("/Serial Poisson/Grid/Refinement")
    , finite_element("/Serial Poisson/", "u", "FE_Q(1)")
    , dof_handler(triangulation)
    , inverse_operator("/Serial Poisson/Solver")
    , preconditioner("/Serial Poisson/Solver/AMG Preconditioner")
    , constants("/Serial Poisson/Constants",
                {"kappa"},
                {1.0},
                {"Diffusion coefficient"})
    , forcing_term("/Serial Poisson/Functions",
                   "kappa*8*pi^2*sin(2*pi*x)*sin(2*pi*y)",
                   "Forcing term")
    , exact_solution("/Serial Poisson/Functions",
                     "sin(2*pi*x)*sin(2*pi*y)",
                     "Exact solution")
    , boundary_conditions("/Serial Poisson/Boundary conditions")
    , data_out("/Serial Poisson/Output")
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
  SerialPoisson<dim, spacedim>::setup_system()
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
    exact_solution().update_user_substitution_map(constants);
    forcing_term().update_user_substitution_map(constants);

    dof_handler.distribute_dofs(finite_element);
    mapping = get_default_linear_mapping(triangulation).clone();

    deallog << "Number of dofs " << dof_handler.n_dofs() << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    boundary_conditions.check_consistency(triangulation);
    boundary_conditions.apply_boundary_conditions(dof_handler, constraints);
    constraints.close();


    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  template <int dim, int spacedim>
  void
  SerialPoisson<dim, spacedim>::assemble_system()
  {
    deallog << "Assemble system" << std::endl;
    const ReferenceCell cell_type = finite_element().reference_cell();

    // TODO: make sure we work also for wedges and pyramids
    const ReferenceCell face_type = cell_type.face_reference_cell(0);

    const Quadrature<dim> quadrature_formula =
      cell_type.get_gauss_type_quadrature<dim>(
        finite_element().tensor_degree() + 1);

    const Quadrature<dim - 1> face_quadrature_formula =
      face_type.get_gauss_type_quadrature<dim - 1>(
        finite_element().tensor_degree() + 1);

    FEValues<dim, spacedim> fe_values(*mapping,
                                      finite_element,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);

    FEFaceValues<dim, spacedim> fe_face_values(*mapping,
                                               finite_element,
                                               face_quadrature_formula,
                                               update_values |
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
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (constants["kappa"] *
                   fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              forcing_term().value(
                                fe_values.quadrature_point(q_index)) * // f(x_q)
                              fe_values.JxW(q_index));                 // dx
          }

        // if (cell->at_boundary())
        //   //  for(const auto face: cell->face_indices())
        //   for (const unsigned int f : cell->face_indices())
        //     if (neumann_ids.find(cell->face(f)->boundary_id()) !=
        //         neumann_ids.end())
        //       {
        //         fe_face_values.reinit(cell, f);
        //         for (const unsigned int q_index :
        //              fe_face_values.quadrature_point_indices())
        //           for (const unsigned int i : fe_face_values.dof_indices())
        //             cell_rhs(i) += fe_face_values.shape_value(i, q_index) *
        //                            neumann_boundary_condition.value(
        //                              fe_face_values.quadrature_point(q_index))
        //                              *
        //                            fe_face_values.JxW(q_index);
        //       }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }



  template <int dim, int spacedim>
  void
  SerialPoisson<dim, spacedim>::solve()
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
  SerialPoisson<dim, spacedim>::output_results(const unsigned cycle) const
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
  SerialPoisson<dim, spacedim>::run()
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
        error_table.error_from_exact(dof_handler, solution, exact_solution());
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

  template class SerialPoisson<2, 2>;
  template class SerialPoisson<3, 3>;
} // namespace PDEs