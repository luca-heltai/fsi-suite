#include "pdes/serial/stokes.h"

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/error_estimator.h>

using namespace dealii;
namespace PDEs
{
  namespace Serial
  {
    template <int dim>
    Stokes<dim>::Stokes()
      : ParameterAcceptor("/Stokes/")
      , component_names(
          ParsedTools::Components::blocks_to_names({"u", "p"}, {dim, 1}))
      , grid_generator("/Stokes/Grid")
      , grid_refinement("/Stokes/Grid/Refinement")
      , finite_element("/Stokes/",
                       component_names,
                       "FESystem[FE_Q(2)^d-FE_Q(1)]")
      , dof_handler(triangulation)
      , inverse_operator("/Stokes/Solver")
      , velocity_preconditioner("/Stokes/Solver/AMG Velocity preconditioner")
      , schur_preconditioner("/Stokes/Solver/AMG Schur preconditioner")
      , constants("/Stokes/Constants", {"eta"}, {1.0}, {"Viscosity"})
      , forcing_term(
          "/Stokes/Functions",
          ParsedTools::Components::join(std::vector<std::string>(dim + 1, "0"),
                                        ";"),
          "Forcing term")
      , exact_solution(
          "/Stokes/Functions",
          ParsedTools::Components::join(std::vector<std::string>(dim + 1, "0"),
                                        ";"),
          "Exact solution")
      , boundary_conditions("/Stokes/Boundary conditions", component_names)
      , error_table(Utilities::split_string_list(component_names),
                    {{VectorTools::H1_norm, VectorTools::L2_norm},
                     {VectorTools::L2_norm}})
      , data_out("/Stokes/Output")
      , velocity(0)
      , pressure(dim)
    {
      enter_subsection("Error table");
      enter_my_subsection(this->prm);
      error_table.add_parameters(this->prm);
      leave_my_subsection(this->prm);
      leave_subsection();
      add_parameter("Console level", this->console_level);
    }



    template <int dim>
    void
    Stokes<dim>::setup_system()
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
      auto blocks = ParsedTools::Components::block_indices(component_names,
                                                           component_names);

      DoFRenumbering::component_wise(this->dof_handler, blocks);

      dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler, blocks);

      mapping = get_default_linear_mapping(triangulation).clone();

      deallog << "Number of dofs " << dof_handler.n_dofs() << "("
              << dofs_per_block[0] << " + " << dofs_per_block[1] << ")"
              << std::endl;

      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      boundary_conditions.check_consistency(triangulation);
      boundary_conditions.apply_essential_boundary_conditions(*mapping,
                                                              dof_handler,
                                                              constraints);
      constraints.close();


      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
      system_matrix.reinit(sparsity_pattern);
      solution.reinit(dofs_per_block);
      system_rhs.reinit(dofs_per_block);
      boundary_conditions.apply_natural_boundary_conditions(
        *mapping, dof_handler, constraints, system_matrix, system_rhs);
    }



    template <int dim>
    void
    Stokes<dim>::assemble_system()
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

      FEValues<dim, dim> fe_values(*mapping,
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
                  const auto &eps_v = fe_values[velocity].gradient(i, q_index);
                  const auto &div_v =
                    fe_values[velocity].divergence(i, q_index);
                  const auto &q = fe_values[pressure].value(i, q_index);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      const auto &eps_u =
                        fe_values[velocity].gradient(j, q_index);
                      const auto &div_u =
                        fe_values[velocity].divergence(j, q_index);
                      const auto &p = fe_values[pressure].value(j, q_index);
                      cell_matrix(i, j) +=
                        (constants["eta"] * scalar_product(eps_v, eps_u) -
                         p * div_v - q * div_u + q * p / constants["eta"]) *
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



    template <int dim>
    void
    Stokes<dim>::solve()
    {
      deallog << "Solve system" << std::endl;

      // SparseDirectUMFPACK A_direct;
      // A_direct.initialize(system_matrix);
      // A_direct.vmult(solution, system_rhs);
      // constraints.distribute(solution);

      velocity_preconditioner.initialize(system_matrix.block(0, 0));
      schur_preconditioner.initialize(system_matrix.block(1, 1));
      deallog << "Preconditioners initialized" << std::endl;

      auto precA =
        linear_operator(system_matrix.block(0, 0), velocity_preconditioner);

      auto precS =
        -1 * linear_operator(system_matrix.block(1, 1), schur_preconditioner);

      // Force compiler to understand what's going on...
      std::array<decltype(precA), 2> tmp_prec = {{precA, precS}};
      const auto diagprecAA = block_diagonal_operator<2>(tmp_prec);

      const auto A    = linear_operator(system_matrix.block(0, 0));
      const auto B    = linear_operator(system_matrix.block(1, 0));
      const auto Bt   = linear_operator(system_matrix.block(0, 1));
      const auto M    = linear_operator(system_matrix.block(1, 1));
      const auto Zero = 0 * M;

      const auto AA = block_operator<2, 2>({{{{A, Bt}}, {{B, Zero}}}});

      // If we use gmres or another non symmetric solver, use a non-symmetric
      // preconditioner
      if (inverse_operator.get_solver_name() != "cg")
        {
          const auto precAA = block_back_substitution(AA, diagprecAA);
          const auto inv    = inverse_operator(AA, precAA);
          solution          = inv * system_rhs;
        }
      else
        {
          const auto inv = inverse_operator(AA, diagprecAA);
          solution       = inv * system_rhs;
        }

      constraints.distribute(solution);

      // const auto A = linear_operator<Vector<double>>(system_matrix.block(0,
      // 0)); const auto invA = inner_inverse_operator(A,
      // velocity_preconditioner);

      // // Build the schur complement
      // const auto Bt =
      //   linear_operator<Vector<double>>(system_matrix.block(0, 1));
      // const auto B = linear_operator<Vector<double>>(system_matrix.block(1,
      // 0)); const auto Mp =
      //   linear_operator<Vector<double>>(system_matrix.block(1, 1));

      // const auto S    = B * invA * Bt;
      // const auto invS = outer_inverse_operator(S, schur_preconditioner);

      // auto &      u = solution.block(0);
      // auto &      p = solution.block(1);
      // const auto &f = system_rhs.block(0);
      // const auto &g = system_rhs.block(1);

      // deallog << "Solving schur complement" << std::endl;

      // p = invS * (B * invA * f - g);
      // deallog << "Computed p (norm = " << p.l2_norm() << ")" << std::endl;

      // deallog << "Solving for velocity" << std::endl;
      // u = invA * (f - Bt * p);
      // deallog << "Computed u (norm = " << u.l2_norm() << ")"
      //         << ")" << std::endl;

      // constraints.distribute(solution);
    }



    template <int dim>
    void
    Stokes<dim>::output_results(const unsigned cycle) const
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



    template <int dim>
    void
    Stokes<dim>::run()
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

    template class Stokes<2>;
    template class Stokes<3>;
  } // namespace Serial
} // namespace PDEs