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

#ifndef inverse_opeator_h
#define inverse_opeator_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_qmrs.h>
#include <deal.II/lac/solver_richardson.h>

namespace ParsedLAC
{
  /**
   * An enum class used to identify different classes derived from
   * SolverControl.
   */
  enum class SolverControlType
  {
    tolerance              = 1 << 0, //!< Use SolverControl
    consecutive_iterations = 1 << 1, //!< Use ConsecutiveControl
    iteration_number       = 1 << 2, //!< Use IterationNumberControl
    reduction              = 1 << 3, //!< Use ReductionControl
  };

  /**
   * A factory that can generate inverse operators according to parameter files.
   *
   * This object is a parsed inverse operator, which uses a parameter file to
   * select a Solver type and SolverControl type.
   *
   * Example usage is the following:
   *
   * @code
   * InverseOperator inverse("/", "cg");
   * ParameterAcceptor::initialize(...);
   *
   * auto Ainv = inverse(linear_operator<VEC>(A), preconditioner);
   *
   * x = Ainv*b;
   * @endcode
   *
   * The parameter file is expected to have the following structure:
   * @code{.sh}
   * set Solver name            = cg
   * set Solver control type    = tolerance
   * set Absolute tolerance     = 1e-12
   * set Relative tolerance     = 1e-12
   * set Maximum iterations     = 1000
   * set Consecutive iterations = 2
   * set Log history            = false
   * set Log result             = false
   * @endcode
   *
   * Every solver control type uses the absolute tolerance, the maximum
   * iterations, and the log parameters. The relative tolerance is used only by
   * the ReductionControl type, i.e., SolverControlType::reduction, while
   * consecutive iterations is used only by the ConsecutiveControl, i.e.,
   * SolverControlType::consecutive_iterations. The special case
   * SolverControlType::iteration_number uses the same parameters of the default
   * one, but it does not fail when reaching the maximum number of iterations.
   * It is thought to be used as an inner solver, for the cases in which you
   * want to apply a fixed number of smoothing iterations, regardless of the
   * reached tolerance.
   */
  class InverseOperator : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Store and parse the parameters that will be needed to create a linear
     * operator that computes the inverse of a matrix via an iterative solver.
     *
     * This object allows you to create an instance of an inverse_operator()
     * LinearOperator, using the selected linear solver and the the selected
     * SolverControl type.
     *
     * @param section_name The name of the section in the parameter file where
     * the pamaeters are stored.  If empty, the section name is the class name.
     * @param default_solver The default solver type.
     * @param control_type The type of solver control to use.
     * @param max_iterations Maximum number of iterations.
     * @param tolerance Absolute tolerance.
     * @param reduction Relative tolerance. Used only if @p control_type is
     * SolverControlType::reduction.
     * @param consecutive_iterations Number of consecutive iterations. Used only
     * if @p control_type is SolverControlType::consecutive_iterations.
     * @param log_history Print the residual at each iteration on deallog.
     * @param log_result Print the number of iterations and the final residual
     * on deallog.
     */
    InverseOperator(
      const std::string       &section_name   = "",
      const std::string       &default_solver = "cg",
      const SolverControlType &control_type   = SolverControlType::tolerance,
      const unsigned int       max_iterations = 1000,
      const double             tolerance      = 1e-12,
      const double             reduction      = 1e-6,
      const unsigned int       consecutive_iterations = 2,
      const bool              &log_history            = false,
      const bool              &log_result             = false);

    /**
     * Create an inverse operator according to the parameters given in the
     * parameter file.
     *
     * You can create many solvers using the same call, by specifying different
     * prefixes each time. If you specify the same prefix, the previous solver
     * stored with that prefix will be destroied.
     *
     * @param op The operator to invert.
     * @param preconditioner The preconditioner to use.
     */
    template <typename Domain,
              typename Payload,
              typename PreconditionerType,
              typename Range = Domain>
    dealii::LinearOperator<Domain, Range, Payload>
    operator()(const dealii::LinearOperator<Range, Domain, Payload> &op,
               const PreconditionerType                             &prec,
               const double       abs_tol = 0.0,
               const std::string &prefix  = "") const;

    /**
     * Create a solver. The solver will use all parameters specified in the
     * parameter file. If abs_tol is non-zero, then a ReductionControl will be
     * used. You can create many solvers using the same call, by specifying
     * different prefixes each time.
     *
     * If you specify the same prefix, the previous solver stored with that
     * prefix will be destroied.
     */
    template <typename Range,
              typename Payload =
                dealii::internal::LinearOperatorImplementation::EmptyPayload,
              typename Domain = Range,
              typename MatrixType,
              typename PreconditionerType>
    dealii::LinearOperator<Range, Domain, Payload>
    solver(const MatrixType         &op,
           const PreconditionerType &prec,
           const double              abs_tol = 0.0,
           const std::string        &prefix  = "") const;


    /**
     * Solve using the specified solver, preconditioner, and tolerance. If the
     * tolerance is non-zero, then a ReductionControl will be used with the
     * given argument, otherwise the parameter file is used to determine all the
     * solver and control parameters.
     */
    template <typename MatrixType,
              typename PreconditionerType,
              typename VectorType>
    void
    solve(const MatrixType         &matrix,
          const PreconditionerType &preconditioner,
          const VectorType         &src,
          VectorType               &dst,
          const double              abs_tol = 0.0) const;


    /**
     * Get the solver name.
     */
    std::string
    get_solver_name() const;

    /**
     * Create a new solver control according to the parameters. If the user
     * supplies a @p abs_tol parameter, the generated SolverControl is a
     * SolverControl with this tolerance.
     */
    std::unique_ptr<dealii::SolverControl>
    setup_new_solver_control(const double abs_tol = 0.0) const;

    /**
     * Create a new solver according to the parameters. If the user supplies a
     * @p abs_tol parameter, the used SolverControl is a SolverControl with
     * this tolerance.
     */
    template <typename Range, typename Payload>
    std::shared_ptr<dealii::SolverBase<Range>>
    setup_new_solver(const double abs_tol = 0.0) const;

  private:
    /**
     * Defines the behaviour of the solver control.
     */
    SolverControlType control_type;

    /**
     * Used internally by the solver.
     */
    mutable std::unique_ptr<dealii::SolverControl> control;

    /**
     * Solver name.
     */
    std::string solver_name;

    /**
     * Default number of maximum iterations required to succesfully
     * complete a solution step.
     */
    unsigned int max_iterations;

    /**
     * Number of consecutive iterations (used only for ConsecutiveControl).
     */
    unsigned int consecutive_iterations;

    /**
     * Default reduction required to succesfully complete a solution
     * step.
     */
    double tolerance;

    /**
     * Default reduction required to succesfully complete a solution
     * step.
     */
    double reduction;

    /**
     * Log the solver history.
     */
    bool log_history;

    /**
     * Log the final result.
     */
    bool log_result;

    /**
     * Local storage for the actual solver object.
     */
    mutable dealii::GeneralDataStorage storage;
  };


#ifndef DOXYGEN
  // ============================================================
  // Explicit template instantiation
  // ============================================================
  template <typename MatrixType,
            typename PreconditionerType,
            typename VectorType>
  void
  InverseOperator::solve(const MatrixType         &matrix,
                         const PreconditionerType &preconditioner,
                         const VectorType         &src,
                         VectorType               &dst,
                         const double              abs_tol) const
  {
    control = setup_new_solver_control(abs_tol);
    if (solver_name == "cg")
      {
        dealii::SolverCG<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "bicgstab")
      {
        dealii::SolverBicgstab<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "gmres")
      {
        dealii::SolverGMRES<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "fgmres")
      {
        dealii::SolverFGMRES<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "minres")
      {
        dealii::SolverMinRes<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "qmrs")
      {
        dealii::SolverQMRS<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else if (solver_name == "richardson")
      {
        dealii::SolverRichardson<VectorType> solver(*control);
        solver.solve(matrix, dst, src, preconditioner);
      }
    else
      {
        Assert(false,
               dealii::ExcInternalError("Solver should not be unknonw."));
      }
  }


  template <typename Range,
            typename Payload,
            typename Domain,
            typename MatrixType,
            typename PreconditionerType>
  dealii::LinearOperator<Range, Domain, Payload>
  InverseOperator::solver(const MatrixType         &op,
                          const PreconditionerType &prec,
                          const double              abs_tol,
                          const std::string        &prefix) const
  {
    control = setup_new_solver_control(abs_tol);

    // Make sure the solver itself is left around until the object is
    // destroyed we need a general storage class, since we have no
    // idea what types are used when calling this function.
    using SolverType = std::shared_ptr<dealii::SolverBase<Range>>;
    auto &solver     = storage.template get_or_add_object_with_name<SolverType>(
      prefix + "solver");

    dealii::LinearOperator<Range, Domain, Payload> inverse;

    auto initialize_solver = [&](auto *s) {
      solver.reset(s);
      inverse = dealii::inverse_operator(op, *s, prec);
    };

    if (solver_name == "cg")
      {
        initialize_solver(new dealii::SolverCG<Range>(*control));
      }
    else if (solver_name == "bicgstab")
      {
        initialize_solver(new dealii::SolverBicgstab<Range>(*control));
      }
    else if (solver_name == "gmres")
      {
        initialize_solver(new dealii::SolverGMRES<Range>(*control));
      }
    else if (solver_name == "fgmres")
      {
        initialize_solver(new dealii::SolverFGMRES<Range>(*control));
      }
    else if (solver_name == "minres")
      {
        initialize_solver(new dealii::SolverMinRes<Range>(*control));
      }
    else if (solver_name == "qmrs")
      {
        initialize_solver(new dealii::SolverQMRS<Range>(*control));
      }
    else if (solver_name == "richardson")
      {
        initialize_solver(new dealii::SolverRichardson<Range>(*control));
      }
    else
      {
        Assert(false,
               dealii::ExcInternalError("Solver should not be unknonw."));
      }
    return inverse;
  }



  template <typename Domain,
            typename Payload,
            typename PreconditionerType,
            typename Range>
  dealii::LinearOperator<Domain, Range, Payload>
  InverseOperator::operator()(
    const dealii::LinearOperator<Range, Domain, Payload> &op,
    const PreconditionerType                             &prec,
    const double                                          abs_tol,
    const std::string                                    &prefix) const
  {
    return solver<Range, Payload>(op, prec, abs_tol, prefix);
  }
#endif
} // namespace ParsedLAC


#endif
