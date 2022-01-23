
#include "parsed_lac/inverse_operator.h"

using namespace dealii;

namespace ParsedLAC
{
  InverseOperator::InverseOperator(const std::string &section_name,
                                   const std::string &default_solver,
                                   const unsigned int default_max_iter,
                                   const double       default_tolerance,
                                   const double       default_reduction)
    : ParameterAcceptor(section_name)
    , solver_name(default_solver)
    , max_iterations(default_max_iter)
    , tolerance(default_tolerance)
    , reduction(default_reduction)
  {
    add_parameter("Solver name",
                  solver_name,
                  "Name of the solver to use. One of cg,bicgstab,gmres,fgmres,"
                  "minres,qmrs, or richardson.",
                  dealii::ParameterAcceptor::prm,
                  dealii::Patterns::Selection("cg|bicgstab|gmres|fgmres|"
                                              "minres|qmrs|richardson"));
    add_parameter("Maximum iterations", max_iterations);
    add_parameter("Absolute tolerance", tolerance);
    add_parameter("Relative tolerance", reduction);
  }

  std::string
  InverseOperator::get_solver_name() const
  {
    return solver_name;
  }

} // namespace ParsedLAC
