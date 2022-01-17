#include "parsed_lac/jacobi.h"

#ifdef DEAL_II_WITH_TRILINOS

using namespace dealii;

namespace ParsedLAC
{
  JacobiPreconditioner::JacobiPreconditioner(const std::string & name,
                                             const double &      omega,
                                             const double &      min_diagonal,
                                             const unsigned int &n_sweeps)
    : ParameterAcceptor(name)
    , PreconditionJacobi()
    , omega(omega)
    , min_diagonal(min_diagonal)
    , n_sweeps(n_sweeps)
  {
    add_parameters();
  }

  void
  JacobiPreconditioner::add_parameters()
  {
    add_parameter(
      "Omega",
      omega,
      "This specifies the relaxation parameter in the Jacobi preconditioner.");
    add_parameter(
      "Min Diagonal",
      min_diagonal,
      "This specifies the minimum value the diagonal elements should "
      "have. This might be necessary when the Jacobi preconditioner is used "
      "on matrices with zero diagonal elements. In that case, a straight- "
      "forward application would not be possible since we would divide by "
      "zero.");
    add_parameter(
      "Number of sweeps",
      n_sweeps,
      "Sets how many times the given operation should be applied during the "
      "vmult() operation.");
  }

  template <typename Matrix>
  void
  JacobiPreconditioner::initialize_preconditioner(const Matrix &matrix)
  {
    TrilinosWrappers::PreconditionJacobi::AdditionalData data;

    data.omega        = omega;
    data.min_diagonal = min_diagonal;
    data.n_sweeps     = n_sweeps;
    this->initialize(matrix, data);
  }
} // namespace ParsedLAC

template void
ParsedLAC::JacobiPreconditioner::initialize_preconditioner<
  dealii::TrilinosWrappers::SparseMatrix>(
  const dealii::TrilinosWrappers::SparseMatrix &);

#endif
