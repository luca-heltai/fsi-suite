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

#include "pdes/mixed_poisson.h"

#include <deal.II/meshworker/mesh_loop.h>

using namespace dealii;

namespace PDEs
{
  namespace MPI
  {
    template <int dim, int spacedim>
    MixedPoisson<dim, spacedim>::MixedPoisson()
      : LinearProblem<dim, spacedim, LAC::LATrilinos>(
          ParsedTools::Components::blocks_to_names({"u", "p"}, {spacedim, 1}),
          "MixedPoisson")
      , coefficient("/MixedPoisson/Functions", "1", "Diffusion coefficient")
      , schur_inverse_operator("/MixedPoisson/Solver/Schur",
                               "cg",
                               ParsedLAC::SolverControlType::iteration_number,
                               10,
                               1e-12,
                               1e-6)
      , velocity(0)
      , pressure(spacedim)
    {}



    template <int dim, int spacedim>
    void
    MixedPoisson<dim, spacedim>::assemble_system_one_cell(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      ScratchData                                                    &scratch,
      CopyData                                                       &copy)
    {
      auto &cell_matrix = copy.matrices[0];
      auto &cell_rhs    = copy.vectors[0];

      cell->get_dof_indices(copy.local_dof_indices[0]);

      const auto &fe_v = scratch.reinit(cell);
      cell_matrix      = 0;
      cell_rhs         = 0;

      for (const unsigned int &qi : fe_v.quadrature_point_indices())
        for (const unsigned int &i : fe_v.dof_indices())
          {
            const auto &v     = fe_v[velocity].value(i, qi);
            const auto &div_v = fe_v[velocity].divergence(i, qi);
            const auto &q     = fe_v[pressure].value(i, qi);

            for (const unsigned int j : fe_v.dof_indices())
              {
                const auto &u     = fe_v[velocity].value(j, qi);
                const auto &div_u = fe_v[velocity].divergence(j, qi);
                const auto &p     = fe_v[pressure].value(j, qi);

                cell_matrix(i, j) += (u * v + p * div_v + div_u * q) * //
                                     fe_v.JxW(qi);                     // dx
              }
            const auto gq =
              this->forcing_term.value(fe_v.quadrature_point(qi), dim);
            for (const unsigned int i : fe_v.dof_indices())
              cell_rhs(i) -= q * gq *      // q*g(x_q)
                             fe_v.JxW(qi); // dx
          }
    }


    template <int dim, int spacedim>
    void
    MixedPoisson<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(this->timer, "solve");
      const auto M  = linear_operator<VectorType>(this->matrix.block(0, 0));
      const auto Bt = linear_operator<VectorType>(this->matrix.block(0, 1));
      const auto B  = linear_operator<VectorType>(this->matrix.block(1, 0));
      this->mass_preconditioner.initialize(this->matrix.block(0, 0));
      const auto PM_inv =
        linear_operator<VectorType>(M, this->mass_preconditioner);
      const auto M_inv =
        this->mass_inverse_operator(M, this->mass_preconditioner);

      // Pointers to the solution and rhs blocks
      auto &U = this->solution.block(0);
      auto &P = this->solution.block(1);

      auto &F = this->rhs.block(0);
      auto &G = this->rhs.block(1);

      // Schur complement
      const auto S = B * M_inv * Bt;

      // Approximated Schur complement
      const auto Sa = B * PM_inv * Bt;

      // Approximation of the inverse of the Schur complement
      const auto Sa_inv_approx =
        this->schur_inverse_operator(Sa, identity_operator(Sa));

      const auto S_inv = this->inverse_operator(S, Sa_inv_approx);

      P = S_inv * (B * M_inv * F - G);
      U = M_inv * (F - Bt * P);

      this->constraints.distribute(this->solution);
      this->locally_relevant_solution = this->solution;
    }

    template class MixedPoisson<2, 2>;
    template class MixedPoisson<2, 3>;
    template class MixedPoisson<3, 3>;
  } // namespace MPI
} // namespace PDEs