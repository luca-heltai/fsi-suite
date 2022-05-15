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

#include "pdes/distributed_lagrange.h"
#include "pdes/distributed_lagrange_elasticity.h"

#include <deal.II/base/logstream.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/petsc_precondition.h>

#include <fstream>
#include <iostream>

#include "lac_initializer.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, int spacedim, typename LacType>
  DistributedLagrangeElasticity<dim, spacedim, LacType>::DistributedLagrangeElasticity()
    : ParameterAcceptor("Distributed Lagrange")
    , space("u,u","Space") //space("u", "Space")
    , space_cache(space.triangulation)
    , embedded("w,w,lambda,lambda","Embedded") //embedded("w", "Embedded")
    , embedded_cache(embedded.triangulation)
    , coupling("/Coupling",ComponentMask(),ComponentMask({0,0,1,1})) //tutte le coponenti della prima con la seconda della seconda var.
    //, mass_solver("/Mass solver")
    , lambda("/LinearElasticity/Lame coefficients", "1.0", "lambda")
    , mu("/LinearElasticity/Lame coefficients", "1.0", "mu")
    , spacedisplacement(0) // abbiamo un solo elemento finito per entrambi - un oggetto con due componenti
    , embdisplacement(0)
    , embdlm(spacedim)
    // aggiungere nel .h come per linear_elasticity
  {}



  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::generate_grids()
  {
    // genero le mesh per i due domini in questione
    TimerOutput::Scope timer_section(space.timer, "generate_grids_and_fes");
    space.grid_generator.generate(space.triangulation);
    embedded.grid_generator.generate(embedded.triangulation);
    // inizializzo qualcosa relativo al coupling
    coupling.initialize(space_cache,
                        space.dof_handler,
                        space.constraints,
                        embedded_cache,
                        embedded.dof_handler,
                        embedded.constraints);
    // e al raffinamento che voglio vicino all'interfaccia
    // CHIEDERE COSA FA DI PRECISO E CHE OGGETTO E' coupling
    coupling.adjust_grid_refinements(space.triangulation,
                                     embedded.triangulation,
                                     true);
  }

  /* da ora in poi commento tutte le cose di accoppiamento perche' per ora non servono */

  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::setup_system()
  {

    /* Qui si fa il set up del sistema, pero' devo capire come trattare le due componenti di u */

    space.setup_system();
    embedded.setup_system();

    const auto row_indices = space.dof_handler.locally_owned_dofs();
    const auto col_indices = embedded.dof_handler.locally_owned_dofs();

    LAC::Initializer init(row_indices,
                          IndexSet(),
                          space.mpi_communicator,
                          col_indices);

    BlockDynamicSparsityPattern dsp(space.dofs_per_block,
                                embedded.dofs_per_block);

    coupling.assemble_sparsity(dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               row_indices,
                                               space.mpi_communicator,
                                               space.locally_relevant_dofs[0]);

    coupling_matrix.reinit(space.locally_owned_dofs,
                           embedded.locally_owned_dofs,
                           dsp,
                           space.mpi_communicator);
  }



  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::assemble_system()
  {
    {
      TimerOutput::Scope timer_section(space.timer,
                                       "Assemble stiffness system");
      // Stiffness matrix and rhs
      typename LinearProblem<spacedim, spacedim, LacType>::ScratchData scratch(
        *space.mapping,
        space.finite_element(),
        space.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<spacedim, spacedim, LacType>::CopyData copy(
        space.finite_element().n_dofs_per_cell());

      for (const auto &cell : space.dof_handler.active_cell_iterators())
        // questo e' un loop sulle celle, per cui qui dentro faccio l'assemblaggio di una cella
        if (cell->is_locally_owned())
          {
            auto &cell_matrix     = copy.matrices[0];
            auto &cell_rhs        = copy.vectors[0];
            cell_matrix           = 0;
            cell_rhs              = 0;
            const auto &fe_values = scratch.reinit(cell);
            cell->get_dof_indices(copy.local_dof_indices[0]);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  {

                    const auto  x = fe_values.quadrature_point(q_index);
                    const auto &eps_v =
                    fe_values[spacedisplacement].symmetric_gradient(i, q_index);
                    const auto &div_v = fe_values[spacedisplacement].divergence(i, q_index);
                    // = 

                    for (const unsigned int j : fe_values.dof_indices())
                    {

                    /*
                      cell_matrix(i, j) +=
                        (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                         fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                         fe_values.JxW(q_index));           // dx
                    cell_rhs(i) +=
                      (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                       space.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                    */

                      const auto &eps_u = fe_values[spacedisplacement].symmetric_gradient(j, q_index);
                      const auto &div_u =
                      fe_values[spacedisplacement].divergence(j, q_index);
                      cell_matrix(i, j) += (2 * mu.value(x) * eps_v * eps_u +
                                          lambda.value(x) * div_v * div_u) *
                                          fe_values.JxW(q_index); // dx mu.value(x),lambda.value(x)
                    }

                  cell_rhs(i) +=
                    (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                      this->space.forcing_term.value(x,
                      this->space.finite_element() //space.finite_element(), embedded.finite_element()
                      .system_to_component_index(i)
                      .first) * // f(x_q)
                      fe_values.JxW(q_index));             // dx
                  }

              }

            space.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              space.matrix,
              space.rhs);
          }

      space.matrix.compress(VectorOperation::add);
      space.rhs.compress(VectorOperation::add);
    }
    {
      TimerOutput::Scope timer_section(space.timer, "Assemble coupling system");
      coupling_matrix = 0.0;
      coupling.assemble_matrix(coupling_matrix);
      coupling_matrix.compress(VectorOperation::add);
    }
    {
      // Embedded mass matrix and rhs
      typename LinearProblem<dim, spacedim, LacType>::ScratchData scratch(
        *embedded.mapping,
        embedded.finite_element(),
        embedded.cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

      typename LinearProblem<dim, spacedim, LacType>::CopyData copy(
        embedded.finite_element().n_dofs_per_cell());

      for (const auto &cell : embedded.dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto &cell_matrix     = copy.matrices[0];
            auto &cell_rhs        = copy.vectors[0];
            cell_matrix           = 0;
            cell_rhs              = 0;
            const auto &fe_values = scratch.reinit(cell);
            cell->get_dof_indices(copy.local_dof_indices[0]);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  {

                    // = parte elasticity senza "displacement" come argomento
                    const auto  x = fe_values.quadrature_point(q_index);
                    const auto &eps_v =
                    fe_values[embdisplacement].symmetric_gradient(i, q_index);
                    const auto &div_v = fe_values[embdisplacement].divergence(i, q_index);
                    const auto &v = fe_values[embdisplacement].value(i,q_index); // test for displ
                    const auto &q = fe_values[embdlm].value(i,q_index); // test for dlm
                    // = 

                    for (const unsigned int j : fe_values.dof_indices())
                    {

                    /*
                      cell_matrix(i, j) +=
                        (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                         fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                         fe_values.JxW(q_index));           // dx
                    cell_rhs(i) +=
                      (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                       space.forcing_term.value(
                         fe_values.quadrature_point(q_index)) * // f(x_q)
                       fe_values.JxW(q_index));                 // dx
                    */

                      const auto &eps_u = fe_values[embdisplacement].symmetric_gradient(j, q_index);
                      const auto &div_u =
                      fe_values[embdisplacement].divergence(j, q_index);
                      const auto &u = fe_values[embdisplacement].value(j,q_index); // test for displ
                      const auto &dlm = fe_values[embdlm].value(j,q_index); // test for dlm
                      cell_matrix(i, j) += (2 * mu.value(x) * eps_v * eps_u +
                                          lambda.value(x) * div_v * div_u + dlm*u + v*q) *
                                          fe_values.JxW(q_index); // dx mu.value(x),lambda.value(x)
                    }

                  cell_rhs(i) +=
                    (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                      this->embedded.forcing_term.value(x,
                      this->embedded.finite_element()
                      .system_to_component_index(i)
                      .first) * // f(x_q)
                      fe_values.JxW(q_index));             // dx
                  }

              }

            embedded.constraints.distribute_local_to_global(
              copy.matrices[0],
              copy.vectors[0],
              copy.local_dof_indices[0],
              embedded.matrix,
              embedded.rhs);
          }

      embedded.matrix.compress(VectorOperation::add);
      embedded.rhs.compress(VectorOperation::add);
      // The rhs of the Lagrange multiplier as a function to plot
      VectorTools::interpolate(embedded.dof_handler,
                               embedded.forcing_term,
                               embedded.solution);
      embedded.solution.compress(VectorOperation::insert);
    }
  }

  
  // da studiare dopo aver capito l'asssemblaggio
  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(space.timer, "Solve system");

    /*
    using BVec       = typename LacType::BlockVector;
    using Vec        = typename BVec::BlockType;
    using LinOp      = LinearOperator<Vec>;
    using BlockLinOp = BlockLinearOperator<BVec>;

    auto A     = linear_operator<Vec>(space.matrix.block(0, 0));
    auto Bt    = linear_operator<Vec>(coupling_matrix);
    auto B     = transpose_operator(Bt);
    auto A_inv = A;
    auto M     = linear_operator<Vec>(embedded.matrix.block(0, 0));
    auto M_inv = M;

    space.preconditioner.initialize(space.matrix.block(0, 0));
    A_inv = space.inverse_operator(A, space.preconditioner);

    embedded.preconditioner.initialize(embedded.matrix.block(0, 0));
    auto M_prec = linear_operator<Vec>(M, embedded.preconditioner);
    M_inv       = mass_solver(M, M_prec);

    auto &lambda       = embedded.solution.block(0);
    auto &embedded_rhs = embedded.rhs.block(0);
    auto &solution     = space.solution.block(0);
    auto &rhs          = space.rhs.block(0);

    auto S      = B * A_inv * Bt;
    auto S_prec = identity_operator(S);
    auto S_inv  = embedded.inverse_operator(S, M_inv);

    lambda   = S_inv * (B * A_inv * rhs - embedded_rhs);
    solution = A_inv * (rhs - Bt * lambda);
    */

    // Solution of the system with block-tri preconditioner
    auto A     = linear_operator<Vec>(space.matrix.block(0,0));
    auto Ct    = linear_operator<Vec>(coupling_matrix.block(0,1));
    auto C     = transpose_operator(Ct);
    auto B     = linear_operator<Vec>(embedded.matrix.block(0,0));

    //auto A_inv = 
    //auto B_inv = 
    // --------

    /*
    // *** Solution of the system in the non-coupling case ***
    auto &A1 = space.matrix.block(0,0); 
    auto &A2 = embedded.matrix.block(0,0);

    auto &f1 = space.rhs.block(0);
    auto &f2 = embedded.rhs.block(0);

    auto &u1 = space.solution.block(0);
    auto &u2 = embedded.solution.block(0);

    SolverControl            solver_control(1000, 1e-12);
    SolverCG<typename LacType::Vector> cg(solver_control);
    space.preconditioner.initialize(A1);
    cg.solve(A1, u1, f1, space.preconditioner);
    embedded.preconditioner.initialize(A2);
    cg.solve(A2, u2, f2, embedded.preconditioner);
    // ***  ***
    

    // Distribute all constraints.
    embedded.constraints.distribute(u2);
    embedded.locally_relevant_solution = embedded.solution;
    space.constraints.distribute(u1);
    space.locally_relevant_solution = space.solution;
    */
  }



  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::output_results(
    const unsigned int cycle)
  {
    space.output_results(cycle);
    embedded.output_results(cycle);
  }


  /* Questo e' il metodo che fa partire il programma sostanzialmente,
  quindi richiama tutte le funzioni precedenti e poi fa un po' di post-processing */

  template <int dim, int spacedim, typename LacType>
  void
  DistributedLagrangeElasticity<dim, spacedim, LacType>::run()
  {
    deallog.depth_console(space.verbosity_level);
    generate_grids();
    for (const auto &cycle : space.grid_refinement.get_refinement_cycles())
      {
        deallog.push("Cycle " + Utilities::int_to_string(cycle));
        setup_system();
        assemble_system();
        solve();
        space.estimate(space.error_per_cell);
        embedded.estimate(embedded.error_per_cell);
        output_results(cycle);
        if (cycle < space.grid_refinement.get_n_refinement_cycles() - 1)
          {
            space.mark(space.error_per_cell);
            space.refine();
            embedded.triangulation.refine_global(1);
            coupling.adjust_grid_refinements(space.triangulation,
                                             embedded.triangulation,
                                             false);
          }
        deallog.pop();
      }
    if (space.mpi_rank == 0)
      {
        space.error_table.output_table(std::cout);
        embedded.error_table.output_table(std::cout);
      }
  }

  /*template class DistributedLagrangeElasticity<1, 2>;
  template class DistributedLagrangeElasticity<2, 2>;
  template class DistributedLagrangeElasticity<2, 3>;
  template class DistributedLagrangeElasticity<3, 3>;*/

  // template class DistributedLagrangeElasticity<1, 2, LAC::LATrilinos>;
  // template class DistributedLagrangeElasticity<2, 2, LAC::LATrilinos>;
  // template class DistributedLagrangeElasticity<2, 3, LAC::LATrilinos>;
  // template class DistributedLagrangeElasticity<3, 3, LAC::LATrilinos>;

  template class DistributedLagrangeElasticity<1, 2, LAC::LAPETSc>;
  template class DistributedLagrangeElasticity<2, 2, LAC::LAPETSc>;
  template class DistributedLagrangeElasticity<2, 3, LAC::LAPETSc>;
  template class DistributedLagrangeElasticity<3, 3, LAC::LAPETSc>;
} // namespace PDEs
