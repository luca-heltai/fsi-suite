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

#include "pdes/stokes.h"

#include <deal.II/lac/linear_operator_tools.h>

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, class LacType>
  Stokes<dim, LacType>::Stokes()
    : LinearProblem<dim, dim, LacType>(
        ParsedTools::Components::join(std::vector<std::string>(dim, "u"), ",") +
          ",p",
        "Stokes")
    , constants("/Stokes/Constants", {"eta"}, {1.0}, {"Viscosity"})
    , schur_preconditioner("/Stokes/Solver/Schur preconditioner")
    , velocity(0)
    , pressure(dim)
  {
    // Fix first pressure dof to zero
    this->add_constraints_call_back.connect([&]() {
      // search for first pressure dof
      unsigned int first_pressure_dof = 0;
      for (unsigned int i = 0; i < this->finite_element().dofs_per_cell; ++i)
        {
          if (this->finite_element().system_to_component_index(i).first == dim)
            {
              first_pressure_dof = i;
              break;
            }
        }
      std::vector<types::global_dof_index> dof_indices(
        this->finite_element().dofs_per_cell);
      this->dof_handler.begin_active()->get_dof_indices(dof_indices);
      this->constraints.add_line(dof_indices[first_pressure_dof]);
    });
  }



  template <int dim, class LacType>
  void
  Stokes<dim, LacType>::assemble_system_one_cell(
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
              fe_values[velocity].symmetric_gradient(i, q_index);
            const auto &div_v = fe_values[velocity].divergence(i, q_index);
            const auto &q     = fe_values[pressure].value(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                const auto &eps_u =
                  fe_values[velocity].symmetric_gradient(j, q_index);
                const auto &div_u = fe_values[velocity].divergence(j, q_index);
                const auto &p     = fe_values[pressure].value(j, q_index);
                // We assemble also the mass matrix for the pressure, to be
                // used as a preconditioner
                cell_matrix(i, j) +=
                  (constants["eta"] * scalar_product(eps_v, eps_u) - div_v * p -
                   div_u * q - p * q / constants["eta"]) *
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



  template <int dim, class LacType>
  void
  Stokes<dim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(this->timer, "solve");

    using BVec       = typename LacType::BlockVector;
    using Vec        = typename BVec::BlockType;
    using LinOp      = LinearOperator<Vec>;
    using BlockLinOp = BlockLinearOperator<BVec>;

    const auto &m = this->system_block_matrix;

    const auto A    = linear_operator<Vec>(m.block(0, 0));
    const auto Bt   = linear_operator<Vec>(m.block(0, 1));
    const auto B    = linear_operator<Vec>(m.block(1, 0));
    const auto Mp   = linear_operator<Vec>(m.block(1, 1));
    const auto Zero = Mp * 0.0;

    auto AA = block_operator<2, 2, BVec>({{{{A, Bt}}, {{B, Zero}}}});

    // auto AA = block_operator<BVec>(m);
    // AA.block(1, 1) *= 0;


    this->preconditioner.initialize(m.block(0, 0));
    schur_preconditioner.initialize(m.block(1, 1));

    auto precA = linear_operator<Vec>(A, this->preconditioner);
    auto precS = linear_operator<Vec>(Mp, schur_preconditioner);

    std::array<LinOp, 2> diag_ops = {{precA, precS}};
    auto diagprecAA               = block_diagonal_operator<2, BVec>(diag_ops);

    deallog << "Preconditioners initialized" << std::endl;

    // If we use gmres or another non symmetric solver, use a non-symmetric
    // preconditioner
    if (this->inverse_operator.get_solver_name() != "cg")
      {
        const auto precAA    = block_back_substitution(AA, diagprecAA);
        const auto inv       = this->inverse_operator(AA, precAA);
        this->block_solution = inv * this->system_block_rhs;
      }
    else
      {
        const auto inv       = this->inverse_operator(AA, diagprecAA);
        this->block_solution = inv * this->system_block_rhs;
      }

    this->constraints.distribute(this->block_solution);
    this->locally_relevant_block_solution = this->block_solution;
  }

  template class Stokes<2, LAC::LAdealii>;
  template class Stokes<3, LAC::LAdealii>;

  template class Stokes<2, LAC::LATrilinos>;
  template class Stokes<3, LAC::LATrilinos>;
} // namespace PDEs