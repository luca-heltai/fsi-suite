#include "pdes/serial/distributed_lagrange.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim>
    DistributedLagrange<dim, spacedim>::DistributedLagrange()
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
      , schur_inverse_operator("/Solver/Schur")
      , schur_preconditioner("/Solver/Schur AMG")
      , data_out("/Data out/Space", "output/embedded")
      , embedded_data_out("/Data out/Embedded", "output/solution")
    {
      add_parameter("Coupling quadrature order", coupling_quadrature_order);
      add_parameter("Console level", console_level);
      add_parameter("Finite element degree (ambient space)",
                    finite_element_degree);
      add_parameter("Finite element degree (embedded space)",
                    embedded_space_finite_element_degree);
      add_parameter("Finite element degree (configuration)",
                    embedded_configuration_finite_element_degree);
    }



    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::generate_grids_and_fes()
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
    DistributedLagrange<dim, spacedim>::adjust_embedded_grid(
      const bool apply_delta_refinement)
    {
      namespace bgi = boost::geometry::index;
      // Now get a vector of all cell centers of the embedded grid and refine
      // them  untill every cell is of diameter smaller than the space
      // surrounding cells
      std::vector<
        std::tuple<Point<spacedim>, decltype(embedded_grid.begin_active())>>
        centers(embedded_grid.n_active_cells());

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
                      cell->neighbor(face_no)->set_refine_flag();
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
            for (const auto &[space_box, space_cell] :
                 tree | bgi::adaptors::queried(bgi::contains(center)))
              {
                const auto &[p1, p2] =
                  space_mapping->get_bounding_box(cell).get_boundary_points();
                const auto diameter = p1.distance(p2);
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
      double embedding_space_minimal_diameter =
        GridTools::minimal_cell_diameter(space_grid);
      deallog << "Embedding minimal diameter: "
              << embedding_space_minimal_diameter
              << ", embedded maximal diameter: "
              << embedded_space_maximal_diameter << ", ratio: "
              << embedded_space_maximal_diameter /
                   embedding_space_minimal_diameter
              << std::endl;
    }


    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::setup_dofs()
    {
      TimerOutput::Scope timer_section(monitor, "setup_dofs");

      space_dh.distribute_dofs(*space_fe);
      DoFTools::make_hanging_node_constraints(space_dh, constraints);
      boundary_conditions.apply_essential_boundary_conditions(*space_mapping,
                                                              space_dh,
                                                              constraints);
      constraints.close();

      DynamicSparsityPattern dsp(space_dh.n_dofs(), space_dh.n_dofs());
      DoFTools::make_sparsity_pattern(space_dh, dsp, constraints);
      stiffness_sparsity.copy_from(dsp);
      stiffness_matrix.reinit(stiffness_sparsity);
      solution.reinit(space_dh.n_dofs());
      rhs.reinit(space_dh.n_dofs());
      deallog << "Embedding dofs: " << space_dh.n_dofs() << std::endl;

      embedded_dh.distribute_dofs(*embedded_fe);
      lambda.reinit(embedded_dh.n_dofs());
      embedded_rhs.reinit(embedded_dh.n_dofs());
      embedded_value.reinit(embedded_dh.n_dofs());
      deallog << "Embedded dofs: " << embedded_dh.n_dofs() << std::endl;

      boundary_conditions.apply_natural_boundary_conditions(
        *space_mapping, space_dh, constraints, stiffness_matrix, rhs);
    }



    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::setup_coupling()
    {
      TimerOutput::Scope     timer_section(monitor, "Setup coupling");
      QGauss<dim>            quad(coupling_quadrature_order);
      DynamicSparsityPattern dsp(space_dh.n_dofs(), embedded_dh.n_dofs());
      NonMatching::create_coupling_sparsity_pattern(*space_grid_tools_cache,
                                                    space_dh,
                                                    embedded_dh,
                                                    quad,
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
    DistributedLagrange<dim, spacedim>::assemble_system()
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

        // The rhs of the Lagrange multiplier
        VectorTools::create_right_hand_side(embedded_mapping(),
                                            embedded_dh,
                                            embedded_quad,
                                            embedded_value_function,
                                            embedded_rhs);
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
        VectorTools::interpolate(embedded_mapping(),
                                 embedded_dh,
                                 embedded_value_function,
                                 embedded_value);
      }
    }
    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(monitor, "Solve system");
      stiffness_preconditioner.initialize(stiffness_matrix);
      auto K      = linear_operator(stiffness_matrix);
      auto Ct     = linear_operator(coupling_matrix);
      auto C      = transpose_operator(Ct);
      auto K_inv  = stiffness_inverse_operator(K, stiffness_preconditioner);
      auto S      = C * K_inv * Ct;
      auto S_prec = identity_operator(S);
      // SolverCG<Vector<double>> solver_cg(schur_solver_control);
      auto S_inv = schur_inverse_operator(S, S_prec);
      lambda     = S_inv * embedded_rhs;
      solution   = K_inv * Ct * lambda;
      constraints.distribute(solution);
    }



    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::output_results()
    {
      TimerOutput::Scope timer_section(monitor, "Output results");
      data_out.attach_dof_handler(space_dh);
      data_out.add_data_vector(solution, component_names);
      data_out.write_data_and_clear(*space_mapping);

      embedded_data_out.attach_dof_handler(embedded_dh);
      embedded_data_out.add_data_vector(lambda, "lambda");
      embedded_data_out.add_data_vector(embedded_value, "g");
      embedded_data_out.write_data_and_clear(embedded_mapping);
    }



    template <int dim, int spacedim>
    void
    DistributedLagrange<dim, spacedim>::run()
    {
      deallog.depth_console(console_level);
      generate_grids_and_fes();
      setup_dofs();
      setup_coupling();
      assemble_system();
      solve();
      output_results();
    }

    template class DistributedLagrange<1, 2>;
    // template class DistributedLagrange<2, 2>;
    // template class DistributedLagrange<2, 3>;
    // template class DistributedLagrange<3, 3>;
  } // namespace Serial

} // namespace PDEs
