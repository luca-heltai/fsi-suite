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



#ifndef ilu_preconditioner_h
#define ilu_preconditioner_h

#include <deal.II/base/config.h>



#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/base/parameter_acceptor.h>

#  include <deal.II/lac/trilinos_precondition.h>

namespace ParsedLAC
{
  /**
   * A parsed ILU preconditioner which uses parameter files to choose
   * between different options. This object is a
   * TrilinosWrappers::PreconditionILU which can be called in place
   * of the preconditioner.
   */
  class ILUPreconditioner : public dealii::ParameterAcceptor,
                            public dealii::TrilinosWrappers::PreconditionILU
  {
  public:
    /**
     * Constructor. Build the preconditioner of a matrix using ILU.
     */
    ILUPreconditioner(const std::string  &name     = "",
                      const unsigned int &ilu_fill = 0,
                      const double       &ilu_atol = 0.0,
                      const double       &ilu_rtol = 1.0,
                      const unsigned int &overlap  = 0);

    /**
     * Initialize the preconditioner using @p matrix.
     */
    template <typename Matrix>
    void
    initialize_preconditioner(const Matrix &matrix);

    using dealii::TrilinosWrappers::PreconditionILU::initialize;

  private:
    /**
     * Declare preconditioner options.
     */
    void
    add_parameters();

    /**
     * This specifies the amount of additional fill-in elements besides
     * the original sparse matrix structure. If \f$k\f$ is fill, the sparsity
     * pattern of \f$Ak+1\f$ is used for the storage of the result of the
     * Gaussian elimination.
     * This is known as \f$ILU( k)\f$ in the literature. When @p ilu_fill is large,
     * the preconditioner comes closer to a (direct) sparse LU decomposition.
     * @note however, that this will drastically increase the memory requirement,
     *  especially when the preconditioner is used in 3D.
     */
    unsigned int ilu_fill;

    /**
     * The amount of perturbation to add to diagonal entries.
     *
     * This parameter allows perturbation of the diagonal of the matrix,
     * which sometimes can help to get better preconditioners especially
     * in the case of bad conditioning. Before factorization, the diagonal entry
     * \f$a_{ii}\f$ is replaced by \f$\alpha sign(a_{ii})+\beta a_{ii}\f$,
     * where \f$\alpha \geq 0\f$ is the absolute threshold @p ilu_atol and
     * \f$\beta \geq 1\f$  is the relative threshold @p ilu_rtol.
     * The default values \f$(\alpha=1, \beta=1)\f$ therefore use the original,
     * unmodified diagonal entry.
     * Suggested values are in the order of \f$10^{−5}\f$ to \f$10^{−2}\f$ for
     * @p ilu_atol and \f$1.01\f$ for @p ilu_rtol.
     */
    double ilu_atol;

    /**
     * Scaling factor for diagonal entries.
     *
     * This parameter allows perturbation of the diagonal of the matrix,
     * which sometimes can help to get better preconditioners especially
     * in the case of bad conditioning. Before factorization, the diagonal entry
     * \f$a_{ii}\f$ is replaced by \f$\alpha sign(a_{ii})+\beta a_{ii}\f$,
     * where \f$\alpha \geq 0\f$ is the absolute threshold @p ilu_atol and
     * \f$\beta \geq 1\f$  is the relative threshold @p ilu_rtol.
     * The default values \f$(\alpha=1, \beta=1)\f$ therefore use the original,
     * unmodified diagonal entry.
     * Suggested values are in the order of \f$10^{−5}\f$ to \f$10^{−2}\f$ for
     * @p ilu_atol and \f$1.01\f$ for @p ilu_rtol.
     */
    double ilu_rtol;

    /**
     * Overlap between processors.
     *
     * This determines how large the overlap of the local matrix portions on
     * each processor in a parallel application should be. An overlap of 0
     * corresponds to a block diagonal decomposition on each processor, an
     * overlap of 1 will additionally include a row \f$j\f$ if there is a
     * nonzero entry in column \f$j\f$ in one of the own rows. Higher overlap
     * numbers work accordingly in a recursive fashion. Increasing overlap will
     * increase communication and storage cost. According to the IFPACK
     * documentation, an overlap of 1 is often effective and values of more than
     * 3 are rarely needed.
     */
    unsigned int overlap;
  };

} // namespace ParsedLAC

#endif // DEAL_II_WITH_TRILINOS

#endif
