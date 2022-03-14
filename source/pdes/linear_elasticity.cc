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

#include "pdes/linear_elasticity.h"

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, int spacedim, class LacType>
  LinearElasticity<dim, spacedim, LacType>::LinearElasticity()
    : LinearProblem<dim, spacedim, LacType>(
        ParsedTools::Components::join(std::vector<std::string>(spacedim, "u"),
                                      ","),
        "LinearElasticity")
    , lambda("/LinearElasticity/Lame coefficients", "1.0", "lambda")
    , mu("/LinearElasticity/Lame coefficients", "1.0", "mu")
    , displacement(0)
  {}



  template <int dim, int spacedim, class LacType>
  void
  LinearElasticity<dim, spacedim, LacType>::assemble_system_one_cell(
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
          {
            const auto  x = fe_values.quadrature_point(q_index);
            const auto &eps_v =
              fe_values[displacement].symmetric_gradient(i, q_index);
            const auto &div_v = fe_values[displacement].divergence(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                const auto &eps_u =
                  fe_values[displacement].symmetric_gradient(j, q_index);
                const auto &div_u =
                  fe_values[displacement].divergence(j, q_index);
                cell_matrix(i, j) += (mu.value(x) * eps_v * eps_u +
                                      lambda.value(x) * div_v * div_u) *
                                     fe_values.JxW(q_index); // dx
              }

            cell_rhs(i) +=
              (fe_values.shape_value(i, q_index) * // phi_i(x_q)
               this->forcing_term.value(x,
                                        this->finite_element()
                                          .system_to_component_index(i)
                                          .first) * // f(x_q)
               fe_values.JxW(q_index));             // dx
          }
      }
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearElasticity<dim, spacedim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(this->timer, "solve");
    const auto         A =
      linear_operator<VectorType>(this->system_block_matrix.block(0, 0));
    this->preconditioner.initialize(this->system_block_matrix.block(0, 0));
    const auto Ainv = this->inverse_operator(A, this->preconditioner);
    this->block_solution.block(0) = Ainv * this->system_block_rhs.block(0);
    this->constraints.distribute(this->block_solution);
    this->locally_relevant_block_solution = this->block_solution;
  }


  template class LinearElasticity<2, 2, LAC::LAdealii>;
  template class LinearElasticity<2, 3, LAC::LAdealii>;
  template class LinearElasticity<3, 3, LAC::LAdealii>;

  template class LinearElasticity<2, 2, LAC::LATrilinos>;
  template class LinearElasticity<2, 3, LAC::LATrilinos>;
  template class LinearElasticity<3, 3, LAC::LATrilinos>;
} // namespace PDEs