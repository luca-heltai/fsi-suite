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

#include "parsed_tools/enum.h"

#ifdef DEAL_II_WITH_PETSC

#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/lac/sparse_matrix.h>

#  include "parsed_lac/amg_petsc.h"


using namespace dealii;

namespace ParsedLAC
{
  PETScAMGPreconditioner::PETScAMGPreconditioner(
    const std::string   &name,
    const bool           symmetric_operator,
    const double         strong_threshold,
    const double         max_row_sum,
    const unsigned int   aggressive_coarsening_num_levels,
    const bool           output_details,
    const RelaxationType relaxation_type_up,
    const RelaxationType relaxation_type_down,
    const RelaxationType relaxation_type_coarse,
    const unsigned int   n_sweeps_coarse,
    const double         tol,
    const unsigned int   max_iter,
    const bool           w_cycle)
    : ParameterAcceptor(name)
    , PETScWrappers::PreconditionBoomerAMG()
    , symmetric_operator(symmetric_operator)
    , strong_threshold(strong_threshold)
    , max_row_sum(max_row_sum)
    , aggressive_coarsening_num_levels(aggressive_coarsening_num_levels)
    , output_details(output_details)
    , relaxation_type_up(relaxation_type_up)
    , relaxation_type_down(relaxation_type_down)
    , relaxation_type_coarse(relaxation_type_coarse)
    , n_sweeps_coarse(n_sweeps_coarse)
    , tol(tol)
    , max_iter(max_iter)
    , w_cycle(w_cycle)
  {
    add_parameters();
  }

  void
  PETScAMGPreconditioner::add_parameters()
  {
    add_parameter(
      "Symmetric operator",
      symmetric_operator,
      "Set this flag to true if you have a symmetric system matrix and you want "
      "to use a solver which assumes a symmetric preconditioner like CG.");

    add_parameter(
      "Strong threshold",
      strong_threshold,
      "Threshold of when nodes are considered strongly connected. See "
      "HYPRE_BoomerAMGSetStrongThreshold(). Recommended values are 0.25 for 2d "
      "and 0.5 for 3d problems, but it is problem dependent.");

    add_parameter(
      "Max row sum",
      max_row_sum,
      "If set to a value smaller than 1.0 then diagonally dominant parts "
      "of the matrix are treated as having no strongly connected nodes. If "
      "the row sum weighted by the diagonal entry is bigger than the given "
      "value, it is considered diagonally dominant. This feature is turned "
      "of by setting the value to 1.0. This is the default as some matrices "
      "can result in having only diagonally dominant entries and thus no "
      "multigrid levels are constructed. The default in BoomerAMG for this "
      "is 0.9. When you try this, check for a reasonable number of levels "
      "created.");

    add_parameter(
      "Aggressive coarsening num levels",
      aggressive_coarsening_num_levels,
      "Number of levels of aggressive coarsening. Increasing this value "
      "reduces the construction time and memory requirements but may "
      "decrease effectiveness.");

    add_parameter(
      "Output details",
      output_details,
      "Setting this flag to true produces debug output from HYPRE, when the "
      "preconditioner is constructed.");

    add_parameter("Relaxation type up", relaxation_type_up);
    add_parameter("Relaxation type down", relaxation_type_down);
    add_parameter("Relaxation type coarse", relaxation_type_coarse);
    add_parameter("Number of sweeps coarse", n_sweeps_coarse);
    add_parameter("Tolerance", tol);
    add_parameter("Max iterations", max_iter);
    add_parameter("W-cycle", w_cycle);
  }



  void
  PETScAMGPreconditioner::initialize(
    const dealii::PETScWrappers::MatrixBase &matrix)
  {
    dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;

    data.symmetric_operator               = symmetric_operator;
    data.strong_threshold                 = strong_threshold;
    data.max_row_sum                      = max_row_sum;
    data.aggressive_coarsening_num_levels = aggressive_coarsening_num_levels;
    data.output_details                   = output_details;
    data.relaxation_type_up               = relaxation_type_up;
    data.relaxation_type_down             = relaxation_type_down;
    data.relaxation_type_coarse           = relaxation_type_coarse;
    data.n_sweeps_coarse                  = n_sweeps_coarse;
    data.tol                              = tol;
    data.max_iter                         = max_iter;
    data.w_cycle                          = w_cycle;

    this->PETScWrappers::PreconditionBoomerAMG::initialize(matrix, data);
  }
} // namespace ParsedLAC

#endif
