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


#include "pdes/linear_problem.h"

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/linear_operator_tools.h>

#include "lac.h"
#include "lac_initializer.h"

using namespace dealii;

using ParsedTools::Components::join;

namespace PDEs
{
  template <int dim, int spacedim, class LacType>
  LinearProblem<dim, spacedim, LacType>::LinearProblem(
    const std::string &component_names,
    const std::string &problem_name)
    : ParameterAcceptor(problem_name == "" ? "" : "/" + problem_name)
    , component_names(component_names)
    , n_components(ParsedTools::Components::n_components(component_names))
    , problem_name(problem_name)
    , section_name(problem_name == "" ? "" : "/" + problem_name)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , timer(deallog.get_console(),
            TimerOutput::summary,
            TimerOutput::cpu_and_wall_times)
    , evolution_type(EvolutionType::steady_state)
    , grid_generator(section_name + "/Grid")
    , grid_refinement(section_name + "/Grid/Refinement")
    , triangulation(mpi_communicator)
    // typename Triangulation<dim, spacedim>::MeshSmoothing(
    //   Triangulation<dim, spacedim>::smoothing_on_refinement |
    //   Triangulation<dim, spacedim>::smoothing_on_coarsening))
    , finite_element(section_name,
                     component_names,
                     "FESystem[FE_Q(1)^" + std::to_string(n_components) + "]")
    , dof_handler(triangulation)
    , inverse_operator(section_name + "/Solver/System")
    , preconditioner(section_name + "/Solver/System AMG preconditioner")
    , mass_inverse_operator(section_name + "/Solver/Mass")
    , mass_preconditioner(section_name + "/Solver/Mass AMG preconditioner")
    , forcing_term(section_name + "/Functions",
                   join(std::vector<std::string>(n_components, "0"), ";"),
                   "Forcing term")
    , exact_solution(section_name + "/Functions",
                     join(std::vector<std::string>(n_components, "0"), ";"),
                     "Exact solution")
    , initial_value(section_name + "/Functions",
                    join(std::vector<std::string>(n_components, "0"), ";"),
                    "Initial value")
    , boundary_conditions(section_name + "/Boundary conditions",
                          component_names,
                          {{numbers::internal_face_boundary_id}},
                          {"all"},
                          {ParsedTools::BoundaryConditionType::dirichlet},
                          {join(std::vector<std::string>(n_components, "0"),
                                ";")})
    , error_table(section_name + "/Error",
                  Utilities::split_string_list(component_names),
                  std::vector<std::set<VectorTools::NormType>>(
                    ParsedTools::Components::n_blocks(component_names),
                    {VectorTools::H1_norm, VectorTools::L2_norm}))
    , data_out(section_name + "/Output")
    , ark_ode_data(section_name + "/ARKode")
  {
    add_parameter("n_threads",
                  number_of_threads,
                  "Fix number of threads during the execution");
    add_parameter("verbosity",
                  verbosity_level,
                  "Verbosity level used with deallog");
    add_parameter("evolution type",
                  evolution_type,
                  "The type of time evolution to use in the linear problem.");
    enter_subsection("Quasi-static");
    add_parameter("start time", start_time, "Start time of the simulation");
    add_parameter("end time", end_time, "End time of the simulation");
    add_parameter("initial time step",
                  desired_start_step_size,
                  "Initial time step of the simulation");
    leave_subsection();

    advance_time_call_back.connect(
      [&](const auto &time, const auto &, const auto &) {
        boundary_conditions.set_time(time);
        forcing_term.set_time(time);
        exact_solution.set_time(time);
      });
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::setup_system()
  {
    TimerOutput::Scope timer_section(timer, "setup_system");
    deallog << "System setup" << std::endl;
    const auto ref_cells = triangulation.get_reference_cells();
    AssertThrow(
      ref_cells.size() == 1,
      ExcMessage(
        "This program does nots support mixed simplex/hex grid types."));

    // Compatible FE space and grid.
    AssertThrow(finite_element().reference_cell() == ref_cells[0],
                ExcMessage("The finite element must be defined on the same "
                           "cell type as the grid."));

    dof_handler.distribute_dofs(finite_element);

    // Since our code runs both for simplex grids and for hyper-cube grids, we
    // need to make sure that we build the correct mapping for the grid. In
    // this code we actually use a linear mapping, independently on the order
    // of the finite element space.
    mapping = get_default_linear_mapping(triangulation).clone();
    deallog << "Number of dofs " << dof_handler.n_dofs() << std::endl;

    const auto blocks =
      ParsedTools::Components::block_indices(component_names, component_names);
    // renumber dofs in a blockwise manner.
    DoFRenumbering::component_wise(dof_handler, blocks);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, blocks);

    locally_owned_dofs =
      dof_handler.locally_owned_dofs().split_by_block(dofs_per_block);

    IndexSet non_blocked_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            non_blocked_locally_relevant_dofs);
    locally_relevant_dofs =
      non_blocked_locally_relevant_dofs.split_by_block(dofs_per_block);

    deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
            << Patterns::Tools::to_string(dofs_per_block) << ")" << std::endl;

    constraints.clear();
    constraints.reinit(non_blocked_locally_relevant_dofs);
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
    boundary_conditions.apply_essential_boundary_conditions(dof_handler,
                                                            constraints);

    // If necessary, derived functions can add constraints to the system here.
    add_constraints_call_back();
    constraints.close();

    ScopedLACInitializer initializer(dofs_per_block,
                                     locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(n_components, n_components);
    for (unsigned int i = 0; i < n_components; ++i)
      for (unsigned int j = 0; j < n_components; ++j)
        coupling[i][j] = DoFTools::always;
    initializer(sparsity, dof_handler, constraints, coupling);
    initializer(sparsity, matrix);
    if (evolution_type == EvolutionType::transient)
      initializer(sparsity, mass_matrix);

    initializer(solution);
    initializer(rhs);
    initializer.ghosted(locally_relevant_solution);

    error_per_cell.reinit(triangulation.n_active_cells());

    boundary_conditions.apply_natural_boundary_conditions(
      *mapping, dof_handler, constraints, matrix, rhs);
    // Update functions with standard constants
    exact_solution.update_constants({});
    forcing_term.update_constants({});

    // Now call anything else that may be needed from the user side
    setup_system_call_back();
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::assemble_system_one_cell(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &,
    ScratchData &,
    CopyData &)
  {
    Assert(false, ExcPureFunctionCalled());
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::copy_one_cell(const CopyData &copy)
  {
    constraints.distribute_local_to_global(copy.matrices[0],
                                           copy.vectors[0],
                                           copy.local_dof_indices[0],
                                           matrix,
                                           rhs);
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::assemble_system()
  {
    TimerOutput::Scope timer_section(timer, "assemble_system");
    Quadrature<dim>    quadrature_formula =
      ParsedTools::Components::get_cell_quadrature(
        triangulation, finite_element().tensor_degree() + 1);

    Quadrature<dim - 1> face_quadrature_formula =
      ParsedTools::Components::get_face_quadrature(
        triangulation, finite_element().tensor_degree() + 1);

    ScratchData scratch(*mapping,
                        finite_element(),
                        quadrature_formula,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values,
                        face_quadrature_formula,
                        update_values | update_quadrature_points |
                          update_JxW_values);

    CopyData copy(finite_element().n_dofs_per_cell());

    auto worker = [&](const auto &cell, auto &scratch, auto &copy) {
      assemble_system_one_cell(cell, scratch, copy);
    };

    auto copier = [&](const auto &copy) { copy_one_cell(copy); };

    using CellFilter = FilteredIterator<
      typename DoFHandler<dim, spacedim>::active_cell_iterator>;

    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                               dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(),
                               dof_handler.end()),
                    worker,
                    copier,
                    scratch,
                    copy);

    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);

    // We assemble the mass matrix only in the transient case
    if (evolution_type == EvolutionType::transient)
      {
        // Assemble also the mass matrix.
        ScratchData scratch(*mapping,
                            finite_element(),
                            quadrature_formula,
                            update_values | update_JxW_values);

        CopyData copy(finite_element().n_dofs_per_cell());

        auto worker = [&](const auto &cell, auto &scratch, auto &copy) {
          const auto &fev  = scratch.reinit(cell);
          copy.matrices[0] = 0;
          cell->get_dof_indices(copy.local_dof_indices[0]);
          for (const auto &q : fev.quadrature_point_indices())
            for (const auto &i : fev.dof_indices())
              for (const auto &j : fev.dof_indices())
                if (finite_element().system_to_component_index(i).first ==
                    finite_element().system_to_component_index(j).first)
                  copy.matrices[0](i, j) +=
                    fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
        };

        auto copier = [&](const auto &copy) {
          constraints.distribute_local_to_global(copy.matrices[0],
                                                 copy.local_dof_indices[0],
                                                 mass_matrix);
        };

        using CellFilter = FilteredIterator<
          typename DoFHandler<dim, spacedim>::active_cell_iterator>;

        WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   dof_handler.begin_active()),
                        CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   dof_handler.end()),
                        worker,
                        copier,
                        scratch,
                        copy);
        mass_matrix.compress(VectorOperation::add);
      }

    assemble_system_call_back();
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::solve()
  {
    Assert(false, ExcPureFunctionCalled());
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::estimate(
    Vector<float> &error_per_cell) const
  {
    TimerOutput::Scope timer_section(timer, "estimate");
    grid_refinement.estimate_error(*mapping,
                                   dof_handler,
                                   locally_relevant_solution,
                                   error_per_cell);

    error_table.error_from_exact(*mapping,
                                 dof_handler,
                                 locally_relevant_solution,
                                 exact_solution);
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::mark(
    const Vector<float> &error_per_cell)
  {
    TimerOutput::Scope timer_section(timer, "mark");
    grid_refinement.mark_cells(error_per_cell, triangulation);
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::refine()
  {
    TimerOutput::Scope timer_section(timer, "refine");
    // Cells have been marked in the mark() method.
    triangulation.execute_coarsening_and_refinement();
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::output_results(
    const unsigned cycle) const
  {
    TimerOutput::Scope timer_section(timer, "output_results");
    deallog << "Output results" << std::endl;
    // Save each cycle in its own file
    const auto suffix =
      Utilities::int_to_string(cycle,
                               Utilities::needed_digits(
                                 grid_refinement.get_n_refinement_cycles()));
    data_out.attach_dof_handler(dof_handler, suffix);
    data_out.add_data_vector(locally_relevant_solution,
                             component_names,
                             dealii::DataOut<dim, spacedim>::type_dof_data);
    // call any additional call backs
    add_data_vector(data_out);
    data_out.write_data_and_clear(*mapping);

    output_results_call_back();
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::print_system_info() const
  {
    if (mpi_rank == 0)
      deallog.depth_console(verbosity_level);
    else
      deallog.depth_console(0);

    if (number_of_threads != -1 && number_of_threads > 0)
      MultithreadInfo::set_thread_limit(
        static_cast<unsigned int>(number_of_threads));

    deallog << "Running " << problem_name << std::endl
            << "Number of cores         : " << MultithreadInfo::n_cores()
            << std::endl
            << "Number of threads       : " << MultithreadInfo::n_threads()
            << std::endl
            << "Number of MPI processes : " << mpi_size << std::endl
            << "MPI rank of this process: " << mpi_rank << std::endl;
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::run()
  {
    switch (evolution_type)
      {
        case EvolutionType::steady_state:
          run_steady_state();
          break;
        case EvolutionType::quasi_static:
          run_quasi_static();
          break;
        case EvolutionType::transient:
          run_transient();
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::run_quasi_static()
  {
    print_system_info();
    deallog << "Solving quasi-static problem" << std::endl;
    grid_generator.generate(triangulation);
    DiscreteTime time(start_time, end_time, desired_start_step_size);
    unsigned int output_cycle = 0;
    while (time.is_at_end() == false)
      {
        const auto cycle = time.get_step_number();
        const auto t     = time.get_next_time();
        const auto dt    = time.get_next_step_size();
        advance_time_call_back(t, dt, cycle);

        deallog << "Timestep " << cycle << ", time = " << t
                << " , step size = " << dt << std::endl;

        setup_system();
        assemble_system();
        solve();
        if (cycle % output_frequency == 0)
          output_results(output_cycle++);
        time.advance_time();
        advance_time_call_back(time.get_current_time(),
                               time.get_previous_step_size(),
                               cycle);
      }
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::setup_transient(ARKode &arkode)
  {
    arkode.output_step =
      [&](const double, const auto &vector, const auto step) {
        locally_relevant_solution = vector;
        output_results(step);
      };

    arkode.implicit_function = [&](const double t, const auto &y, auto &res) {
      deallog << "Evaluation at time " << t << std::endl;
      advance_time_call_back(t, 0.0, 0);
      matrix.vmult(res, y);
      res.sadd(-1.0, 1.0, rhs);
      return 0;
    };

    arkode.mass_times_vector = [&](const double, const auto &src, auto &dst) {
      mass_matrix.vmult(dst, src);
      return 0;
    };


    arkode.jacobian_times_vector =
      [&](const auto &v, auto &Jv, double, const auto &, const auto &) {
        matrix.vmult(Jv, v);
        Jv *= -1.0;
        return 0;
      };


    arkode.solve_mass =
      [&](auto &op, auto &prec, auto &dst, const auto &src, double tol) -> int {
      try
        {
          deallog << "Solving mass system" << std::endl;
          mass_inverse_operator.solve(op, prec, src, dst, tol);
          return 0;
        }
      catch (...)
        {
          return 1;
        }
    };

    arkode.solve_linearized_system =
      [&](auto &op, auto &prec, auto &dst, const auto &src, double tol) -> int {
      try
        {
          deallog << "Solving linearized system" << std::endl;
          inverse_operator.solve(op, prec, src, dst, tol);
          return 0;
        }
      catch (...)
        {
          return 1;
        }
    };
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::run_transient()
  {
    print_system_info();
    deallog << "Solving transient problem" << std::endl;
    grid_generator.generate(triangulation);
    setup_system();
    assemble_system();

    VectorTools::interpolate(*mapping, dof_handler, initial_value, solution);

    ARKode arkode(ark_ode_data, mpi_communicator);
    setup_transient(arkode);
    setup_arkode_call_back(arkode);

    // Just start the solver.
    auto res = arkode.solve_ode(solution);

    // Check the result.
    AssertThrow(res != 0,
                ExcMessage("ARKode solver failed with error code " +
                           std::to_string(res)));
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::run_steady_state()
  {
    print_system_info();
    deallog << "Solving steady state problem" << std::endl;
    grid_generator.generate(triangulation);
    for (const auto &cycle : grid_refinement.get_refinement_cycles())
      {
        deallog << "Cycle " << cycle << std::endl;
        setup_system();
        assemble_system();
        solve();
        estimate(error_per_cell);
        output_results(cycle);
        if (cycle < grid_refinement.get_n_refinement_cycles() - 1)
          {
            mark(error_per_cell);
            refine();
          }
      }
    if (this->mpi_rank == 0)
      error_table.output_table(std::cout);
  }

  template class LinearProblem<1, 1, LAC::LAdealii>;
  template class LinearProblem<1, 2, LAC::LAdealii>;
  template class LinearProblem<1, 3, LAC::LAdealii>;
  template class LinearProblem<2, 2, LAC::LAdealii>;
  template class LinearProblem<2, 3, LAC::LAdealii>;
  template class LinearProblem<3, 3, LAC::LAdealii>;

  template class LinearProblem<1, 1, LAC::LAPETSc>;
  template class LinearProblem<1, 2, LAC::LAPETSc>;
  template class LinearProblem<1, 3, LAC::LAPETSc>;
  template class LinearProblem<2, 2, LAC::LAPETSc>;
  template class LinearProblem<2, 3, LAC::LAPETSc>;
  template class LinearProblem<3, 3, LAC::LAPETSc>;

  template class LinearProblem<1, 1, LAC::LATrilinos>;
  template class LinearProblem<1, 2, LAC::LATrilinos>;
  template class LinearProblem<1, 3, LAC::LATrilinos>;
  template class LinearProblem<2, 2, LAC::LATrilinos>;
  template class LinearProblem<2, 3, LAC::LATrilinos>;
  template class LinearProblem<3, 3, LAC::LATrilinos>;
} // namespace PDEs
