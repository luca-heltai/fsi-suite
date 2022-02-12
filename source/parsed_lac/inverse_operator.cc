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


#include "parsed_lac/inverse_operator.h"

#include "parsed_tools/enum.h"

using namespace dealii;

namespace ParsedLAC
{
  InverseOperator::InverseOperator(const std::string &      section_name,
                                   const std::string &      solver_name,
                                   const SolverControlType &control_type,
                                   const unsigned int       max_iterations,
                                   const double             tolerance,
                                   const double             reduction,
                                   const unsigned int consecutive_iterations,
                                   const bool &       log_history,
                                   const bool &       log_result)
    : ParameterAcceptor(section_name)
    , control_type(control_type)
    , solver_name(solver_name)
    , max_iterations(max_iterations)
    , consecutive_iterations(consecutive_iterations)
    , tolerance(tolerance)
    , reduction(reduction)
    , log_history(log_history)
    , log_result(log_result)
  {
    add_parameter("Solver name",
                  this->solver_name,
                  "Name of the solver to use. One of cg,bicgstab,gmres,fgmres,"
                  "minres,qmrs, or richardson.",
                  dealii::ParameterAcceptor::prm,
                  dealii::Patterns::Selection("cg|bicgstab|gmres|fgmres|"
                                              "minres|qmrs|richardson"));
    add_parameter("Solver control type", this->control_type);
    add_parameter("Maximum iterations", this->max_iterations);
    add_parameter("Consecutive iterations", this->consecutive_iterations);
    add_parameter("Absolute tolerance", this->tolerance);
    add_parameter("Relative tolerance", this->reduction);
    add_parameter("Log history", this->log_history);
    add_parameter("Log result", this->log_result);
  }

  std::string
  InverseOperator::get_solver_name() const
  {
    return solver_name;
  }


  std::unique_ptr<dealii::SolverControl>
  InverseOperator::setup_new_solver_control() const
  {
    std::unique_ptr<dealii::SolverControl> result;
    switch (control_type)
      {
        case SolverControlType::tolerance:
          result.reset(new SolverControl(
            max_iterations, tolerance, log_history, log_result));
          break;
        case SolverControlType::reduction:
          result.reset(new ReductionControl(
            max_iterations, tolerance, reduction, log_history, log_result));
          break;
        case SolverControlType::consecutive_iterations:
          result.reset(new ConsecutiveControl(max_iterations,
                                              tolerance,
                                              consecutive_iterations,
                                              log_history,
                                              log_result));
          break;
        case SolverControlType::iteration_number:
          result.reset(new IterationNumberControl(
            max_iterations, tolerance, log_history, log_result));
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
    return result;
  }

} // namespace ParsedLAC
