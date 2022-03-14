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
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
    , grid_generator(section_name + "/Grid")
    , grid_refinement(section_name + "/Grid/Refinement")
    , triangulation(mpi_communicator,
                    typename Triangulation<dim, spacedim>::MeshSmoothing(
                      Triangulation<dim, spacedim>::smoothing_on_refinement |
                      Triangulation<dim, spacedim>::smoothing_on_coarsening))
    , finite_element(section_name,
                     component_names,
                     "FESystem[FE_Q(1)^" + std::to_string(n_components) + "]")
    , dof_handler(triangulation)
    , inverse_operator(section_name + "/Solver")
    , preconditioner(section_name + "/Solver/AMG preconditioner")
    , forcing_term(section_name + "/Functions",
                   join(std::vector<std::string>(n_components, "0"), ";"),
                   "Forcing term")
    , exact_solution(section_name + "/Functions",
                     join(std::vector<std::string>(n_components, "0"), ";"),
                     "Exact solution")
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
  {
    add_parameter("n_threads",
                  number_of_threads,
                  "Fix number of threads during the execution");
    add_parameter("verbosity",
                  verbosity_level,
                  "Verbosity level used with deallog");
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::setup_system()
  {
    TimerOutput::Scope timer_section(timer, "setup_system");
    pcout << "System setup" << std::endl;
    const auto ref_cells = triangulation.get_reference_cells();
    AssertThrow(
      ref_cells.size() == 1,
      ExcMessage(
        "This program does nots support mixed simplx/hex grid types."));

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
    pcout << "Number of dofs " << dof_handler.n_dofs() << std::endl;

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
    boundary_conditions.apply_essential_boundary_conditions(dof_handler,
                                                            constraints);

    // If necessary, derived functions can add constraints to the system here.
    add_constraints_call_back();
    constraints.close();

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

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << Patterns::Tools::to_string(dofs_per_block) << ")" << std::endl;

    constraints.clear();
    constraints.reinit(non_blocked_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    boundary_conditions.apply_essential_boundary_conditions(*mapping,
                                                            dof_handler,
                                                            constraints);
    constraints.close();

    ScopedLACInitializer initializer(dofs_per_block,
                                     locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(n_components, n_components);
    for (unsigned int i = 0; i < n_components; ++i)
      for (unsigned int j = 0; j < n_components; ++j)
        coupling[i][j] = DoFTools::always;
    initializer(system_block_sparsity, dof_handler, constraints, coupling);
    initializer(system_block_sparsity, system_block_matrix);

    initializer(block_solution);
    initializer(system_block_rhs);
    initializer.ghosted(locally_relevant_block_solution);

    error_per_cell.reinit(triangulation.n_active_cells());

    boundary_conditions.apply_natural_boundary_conditions(*mapping,
                                                          dof_handler,
                                                          constraints,
                                                          system_block_matrix,
                                                          system_block_rhs);
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
                                           system_block_matrix,
                                           system_block_rhs);
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

    system_block_matrix.compress(VectorOperation::add);
    system_block_rhs.compress(VectorOperation::add);

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
                                   locally_relevant_block_solution,
                                   error_per_cell);

    error_table.error_from_exact(*mapping,
                                 dof_handler,
                                 locally_relevant_block_solution,
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
    data_out.add_data_vector(locally_relevant_block_solution, component_names);
    // call any additional call backs
    add_data_vector(data_out);
    data_out.write_data_and_clear(*mapping);
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::print_system_info() const
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      deallog.depth_console(verbosity_level);
    else
      deallog.depth_console(0);

    if (number_of_threads != -1 && number_of_threads > 0)
      MultithreadInfo::set_thread_limit(
        static_cast<unsigned int>(number_of_threads));

    pcout << "Running " << problem_name << std::endl
          << "Number of cores         : " << MultithreadInfo::n_cores()
          << std::endl
          << "Number of threads       : " << MultithreadInfo::n_threads()
          << std::endl
          << "Number of MPI processes : "
          << Utilities::MPI::n_mpi_processes(mpi_communicator) << std::endl
          << "MPI rank of this process: "
          << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearProblem<dim, spacedim, LacType>::run()
  {
    print_system_info();
    grid_generator.generate(triangulation);
    for (const auto &cycle : grid_refinement.get_refinement_cycles())
      {
        pcout << "Cycle " << cycle << std::endl;
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
    if (pcout.is_active())
      error_table.output_table(std::cout);
  }

  template class LinearProblem<1, 1, LAC::LAdealii>;
  template class LinearProblem<1, 2, LAC::LAdealii>;
  template class LinearProblem<2, 2, LAC::LAdealii>;
  template class LinearProblem<2, 3, LAC::LAdealii>;
  template class LinearProblem<3, 3, LAC::LAdealii>;

  // // Explicit instantiation: no one dimensional parallel
  // // triangulation
  // template class LinearProblem<2, 2, LAPETSc>;
  // template class LinearProblem<2, 3, LAPETSc>;
  // template class LinearProblem<3, 3, LAPETSc>;

  // Explicit instantiation: no one dimensional parallel
  // triangulation
  template class LinearProblem<2, 2, LAC::LATrilinos>;
  template class LinearProblem<2, 3, LAC::LATrilinos>;
  template class LinearProblem<3, 3, LAC::LATrilinos>;
} // namespace PDEs
