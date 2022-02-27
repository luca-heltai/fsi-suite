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

#include "pdes/mpi/poisson.h"

using namespace dealii;

namespace PDEs
{
  namespace MPI
  {
    template <int dim, int spacedim>
    Poisson<dim, spacedim>::Poisson()
      : LinearProblem<dim, spacedim>("u", "Poisson")
      , coefficient("/Poisson/Functions", "1", "Diffusion coefficient")
    {}



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::assemble_system_one_cell(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      ScratchData &                                                   scratch,
      CopyData &                                                      copy)
    {
      auto &cell_matrix = copy.matrices[0];
      auto &cell_rhs    = copy.vectors[0];

      cell->get_dof_indices(copy.local_dof_indices[0]);

      const auto &fe_values = scratch.reinit(cell);
      cell_matrix           = 0;
      cell_rhs              = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (coefficient.value(
                   fe_values.quadrature_point(q_index)) * // a(x_q)
                 fe_values.shape_grad(i, q_index) *       // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) *       // grad phi_j(x_q)
                 fe_values.JxW(q_index));                 // dx
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            this->forcing_term.value(
                              fe_values.quadrature_point(q_index)) * // f(x_q)
                            fe_values.JxW(q_index));                 // dx
        }
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(this->timer, "solve");
      const auto         A =
        linear_operator<LA::MPI::Vector>(this->system_block_matrix.block(0, 0));
      this->preconditioner.initialize(this->system_block_matrix.block(0, 0));
      const auto Ainv = this->inverse_operator(A, this->preconditioner);
      this->block_solution.block(0) = Ainv * this->system_block_rhs.block(0);
      this->constraints.distribute(this->block_solution);
      this->locally_relevant_block_solution = this->block_solution;
    }


    template class Poisson<2, 2>;
    template class Poisson<2, 3>;
    template class Poisson<3, 3>;
  } // namespace MPI
} // namespace PDEs