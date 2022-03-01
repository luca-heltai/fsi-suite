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

#include "pdes/mpi/linear_elasticity.h"

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  namespace MPI
  {
    template <int dim>
    LinearElasticity<dim>::LinearElasticity()
      : LinearProblem<dim>(
          ParsedTools::Components::join(std::vector<std::string>(dim, "u"),
                                        ","),
          "LinearElasticity")
      , coefficient("/LinearElasticity/Functions", "1", "Diffusion coefficient")
      , constants("/LinearElasticity/Constants",
                  {"lambda", "mu"},
                  {1.0, 1.0},
                  {"Lame parameter", "Lame parameter"})
      , displacement(0)
    {}



    template <int dim>
    void
    LinearElasticity<dim>::assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      CopyData &                                            copy)
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

                  cell_matrix(i, j) += (constants["mu"] * eps_v * eps_u +
                                        constants["lambda"] * div_v * div_u) *
                                       fe_values.JxW(q_index); // dx
                }

              cell_rhs(i) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 this->forcing_term.value(fe_values.quadrature_point(q_index),
                                          this->finite_element()
                                            .system_to_component_index(i)
                                            .first) * // f(x_q)
                 fe_values.JxW(q_index));             // dx
            }
        }
    }



    template <int dim>
    void
    LinearElasticity<dim>::solve()
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


    template class LinearElasticity<2>;
    template class LinearElasticity<3>;
  } // namespace MPI
} // namespace PDEs