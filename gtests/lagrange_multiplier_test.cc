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

#include <deal.II/base/config.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "pdes/lagrange_multipliers.h"

using namespace dealii;
namespace PDEs
{

  TEST(LagrangeMultipliers, Basic_IDS_MPI)
  {
    static const int                    dim = 2;
    PDEs::MPI::LagrangeMultipliers<dim> lm;
    ParameterAcceptor::initialize(FSI_SUITE_SOURCE_DIR
                                  "/prms/mpi_babuska_2d.prm",
                                  "mpi_babuska_2d.prm");
    lm.generate_grids();
    lm.setup_system();
    lm.assemble_system();

    {
      using BVec = typename LAC::LATrilinos::BlockVector;
      using Vec  = BVec::BlockType;

      using LinOp      = LinearOperator<Vec>;
      using BlockLinOp = BlockLinearOperator<BVec>;

      auto A     = linear_operator<Vec>(lm.space.matrix.block(0, 0));
      auto Bt    = linear_operator<Vec>(lm.coupling_matrix);
      auto B     = transpose_operator(Bt);
      auto A_inv = A;
      auto M     = linear_operator<Vec>(lm.embedded.matrix.block(0, 0));
      auto M_inv = M;

      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes(lm.space.dof_handler,
                                       ComponentMask({true}),
                                       constant_modes);
      lm.space.preconditioner.set_constant_modes(constant_modes);
      lm.space.preconditioner.initialize(lm.space.matrix.block(0, 0));
      A_inv = lm.space.inverse_operator(A, lm.space.preconditioner);

      lm.embedded.preconditioner.initialize(lm.embedded.matrix.block(0, 0));
      auto M_prec = linear_operator<Vec>(M, lm.embedded.preconditioner);
      M_inv       = lm.mass_solver(M, M_prec);

      // auto M_prec = B * A * Bt + M;

      auto &lambda       = lm.embedded.solution.block(0);
      auto &embedded_rhs = lm.embedded.rhs.block(0);
      auto &solution     = lm.space.solution.block(0);
      auto &rhs          = lm.space.rhs.block(0);

      // Start with some trivial checks.
      VectorTools::interpolate(lm.space.dof_handler,
                               lm.space.exact_solution,
                               solution);
      VectorTools::interpolate(lm.embedded.dof_handler,
                               lm.space.exact_solution,
                               lambda);

      // Now these two contain the same values evaluated on different support
      // points. First we check we have nonzero vectors.
      ASSERT_GT(solution.l2_norm(), 0.0);
      ASSERT_GT(lambda.l2_norm(), 0.0);

      deallog << "Solution norm: " << solution.l2_norm() << std::endl;
      deallog << "Lambda norm: " << lambda.l2_norm() << std::endl;

      // Check that the entries are non zero
      embedded_rhs = M * lambda;
      ASSERT_GT(embedded_rhs.l2_norm(), 0.0);
      deallog << "|| M*lambda ||: " << embedded_rhs.l2_norm() << std::endl;

      // Check that the entries are non zero
      embedded_rhs = B * solution;
      ASSERT_GT(embedded_rhs.l2_norm(), 0.0);
      deallog << "|| B*u ||: " << embedded_rhs.l2_norm() << std::endl;

      // It should be (if the spaces are
      // compatible), that the following holds: B * solution = M * lambda.
      embedded_rhs = M * lambda - B * solution;
      ASSERT_NEAR(embedded_rhs.l2_norm(), 0.0, 1e-10);
      deallog << "|| M*lambda - B*u ||: " << embedded_rhs.l2_norm()
              << std::endl;

      // Now we can solve the system.
      VectorTools::interpolate(lm.embedded.dof_handler,
                               lm.space.exact_solution,
                               lambda);
      embedded_rhs = M * lambda;

      auto ZeroM = null_operator(M);

      const auto AA =
        block_operator<2, 2, BVec, BVec>({{{{A, Bt}}, {{B, ZeroM}}}});

      const auto AA_diag_inv = block_diagonal_operator<2, BVec, BVec>(
        std::array<LinOp, 2>{{A_inv, M_inv}});

      BVec temp_rhs, temp_sol;
      AA.reinit_domain_vector(temp_sol, true);
      AA.reinit_range_vector(temp_rhs, true);

      temp_rhs.block(0) = rhs;
      temp_rhs.block(1) = embedded_rhs;

      deallog << "Block RHS norm: " << temp_rhs.l2_norm() << std::endl;

      auto tri_back = block_back_substitution(AA, AA_diag_inv);
      auto tri_for  = block_forward_substitution(AA, AA_diag_inv);

      auto AA_inv = lm.embedded.inverse_operator(AA, AA_diag_inv);
      // auto AA_inv = lm.embedded.inverse_operator(AA, tri_for);
      // auto AA_inv = lm.embedded.inverse_operator(AA, tri_back);

      temp_sol = AA_inv * temp_rhs;
      deallog << "Block solution norm: " << temp_sol.l2_norm() << std::endl;

      solution = temp_sol.block(0);
      lambda   = temp_sol.block(1);

      // // auto P = mean_value_filter(A_inv);
      // auto P = identity_operator(A_inv);
      // P      = P - Bt * M_inv * B;

      // A_inv = P * A_inv * P;

      // auto S      = B * A_inv * Bt;
      // auto S_prec = identity_operator(S);
      // auto S_inv  = lm.embedded.inverse_operator(S, M_inv);


      // lambda   = S_inv * (B * A_inv * rhs - embedded_rhs);
      // solution = A_inv * (rhs - Bt * lambda);

      // Write again the exact solution on the embedded space (not lambda)
      VectorTools::interpolate(lm.embedded.dof_handler,
                               lm.space.exact_solution,
                               lambda);

      // // Distribute all constraints.
      lm.embedded.constraints.distribute(lambda);
      lm.embedded.locally_relevant_solution = lm.embedded.solution;
      lm.space.constraints.distribute(solution);
      lm.space.locally_relevant_solution = lm.space.solution;

      lm.output_results(0);

      // Check that the solution is correct on the boundary. This is now
      // *after *
      // solving the system.
      // lambda = M * embedded_rhs - B * solution;
      // ASSERT_NEAR(lambda.l2_norm(), 0.0, 1e-10);
    }
  }
} // namespace PDEs