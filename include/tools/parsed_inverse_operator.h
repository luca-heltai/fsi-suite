#ifndef parsed_inverse_opeator_h
#define parsed_inverse_opeator_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_qmrs.h>
#include <deal.II/lac/solver_richardson.h>

namespace Tools
{
  /**
   * A factory that can generate inverse operators according to parameter files.
   *
   * This object is a parsed inverse operator, which uses a parameter file to
   * select a solver type, maximum iterators, an absolute tolerance, and a
   * relative tolerance.
   *
   * Example usage is the following:
   *
   * @code
   * ParsedInverseOperator inverse("/", "cg", 100, 1e-12, 1e-12);
   * ParameterAcceptor::initialize(...);
   *
   * auto Ainv = inverse(linear_operator<VEC>(A), preconditioner);
   *
   * x = Ainv*b;
   * @endcode
   */
  class ParsedInverseOperator : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Store the parameters that will be needed to create a linear operator.
     *
     * A section name can be specified, the solver type, the
     * maximum number of iterations, and the reduction to reach
     * convergence. If you know in advance the operators this object
     * will need, you can also supply them here. They default to the
     * identity, and you can assign them later by setting op and prec.
     */
    ParsedInverseOperator(const std::string &section_name   = "",
                          const std::string &default_solver = "cg",
                          const unsigned int max_iter       = 1000,
                          const double       tolerance      = 1e-12,
                          const double       reduction      = 1e-12);

    /**
     * Create an inverse operator according to the parameters given in the
     * parameter file.
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
               const PreconditionerType &prec) const;

  private:
    /**
     * ReductionControl. Used internally by the solver.
     */
    mutable dealii::ReductionControl control;

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
     * Default reduction required to succesfully complete a solution
     * step.
     */
    double tolerance;

    /**
     * Default reduction required to succesfully complete a solution
     * step.
     */
    double reduction;
  };

  // ============================================================
  // Explicit template instantiation
  // ============================================================
  template <typename Domain,
            typename Payload,
            typename PreconditionerType,
            typename Range>
  dealii::LinearOperator<Domain, Range, Payload>
  ParsedInverseOperator::operator()(
    const dealii::LinearOperator<Range, Domain, Payload> &op,
    const PreconditionerType &                            prec) const
  {
    control.set_max_steps(max_iterations);
    control.set_reduction(reduction);
    control.set_tolerance(tolerance);

    // Make sure this is left around until the object is destroyed
    static std::unique_ptr<dealii::SolverBase<Range>> solver;
    dealii::LinearOperator<Domain, Range, Payload>    inverse;

    auto initialize_solver = [&](auto *s) {
      solver.reset(s);
      inverse = dealii::inverse_operator(op, *s, prec);
    };

    if (solver_name == "cg")
      {
        initialize_solver(new dealii::SolverCG<Range>(control));
      }
    else if (solver_name == "bicgstab")
      {
        initialize_solver(new dealii::SolverBicgstab<Range>(control));
      }
    else if (solver_name == "gmres")
      {
        initialize_solver(new dealii::SolverGMRES<Range>(control));
      }
    else if (solver_name == "fgmres")
      {
        initialize_solver(new dealii::SolverFGMRES<Range>(control));
      }
    else if (solver_name == "minres")
      {
        initialize_solver(new dealii::SolverMinRes<Range>(control));
      }
    else if (solver_name == "qmrs")
      {
        initialize_solver(new dealii::SolverQMRS<Range>(control));
      }
    else if (solver_name == "richardson")
      {
        initialize_solver(new dealii::SolverRichardson<Range>(control));
      }
    else
      {
        Assert(false,
               dealii::ExcInternalError("Solver should not be unknonw."));
      }
    return inverse;
  }
} // namespace Tools


#endif
