//-----------------------------------------------------------
//
//    Copyright (C) 2015 by the deal2lkit authors
//
//    This file is part of the deal2lkit library.
//
//    The deal2lkit library is free software; you can use it, redistribute
//    it, and/or modify it under the terms of the GNU Lesser General
//    Public License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//    The full text of the license can be found in the file LICENSE at
//    the top level of the deal2lkit distribution.
//
//-----------------------------------------------------------

#ifndef jacobi_preconditioner_h
#define jacobi_preconditioner_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/base/parameter_acceptor.h>

#  include <deal.II/lac/trilinos_precondition.h>


namespace ParsedLAC
{
  /**
   * A parsed Jacobi preconditioner which uses parameter files to choose
   * between different options. This object is a
   * TrilinosWrappers::PreconditionJacobi which can be called in place
   * of the preconditioner.
   */
  class JacobiPreconditioner
    : public dealii::ParameterAcceptor,
      public dealii::TrilinosWrappers::PreconditionJacobi
  {
  public:
    /**
     * Constructor. Build the preconditioner of a matrix using Jacobi.
     */
    JacobiPreconditioner(const std::string  &name         = "",
                         const double       &omega        = 1,
                         const double       &min_diagonal = 0,
                         const unsigned int &n_sweeps     = 1);

    /**
     * Initialize the preconditioner using @p matrix.
     */
    template <typename Matrix>
    void
    initialize_preconditioner(const Matrix &matrix);

    using dealii::TrilinosWrappers::PreconditionJacobi::initialize;

  private:
    /**
     * Declare preconditioner options.
     */
    void
    add_parameters();

    /**
     * This specifies the relaxation parameter in the Jacobi
     * preconditioner.
     */
    double omega;

    /**
     * This specifies the minimum value the diagonal elements should
     * have. This might be necessary when the Jacobi preconditioner is
     * used on matrices with zero diagonal elements. In that case, a
     * straight- forward application would not be possible since we
     * would divide by zero.
     */
    double min_diagonal;

    /**
     * Sets how many times the given operation should be applied during
     * the vmult() operation.
     */
    unsigned int n_sweeps;
  };

} // namespace ParsedLAC

#endif // DEAL_II_WITH_TRILINOS

#endif
