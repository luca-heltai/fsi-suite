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

#ifndef amg_petsc_preconditioner_h
#define amg_petsc_preconditioner_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PETSC

#  include <deal.II/base/parameter_acceptor.h>

#  include <deal.II/lac/petsc_precondition.h>

namespace ParsedLAC
{
  /**
   * A parsed AMG preconditioner which uses parameter files to choose
   * between different options. This object is a
   * TrilinosWrappers::PreconditionAMG which can be called in place of
   * the preconditioner.
   */
  class PETScAMGPreconditioner
    : public dealii::ParameterAcceptor,
      public dealii::PETScWrappers::PreconditionBoomerAMG
  {
  public:
    using RelaxationType = dealii::PETScWrappers::PreconditionBoomerAMG::
      AdditionalData::RelaxationType;
    /**
     * Constructor. Build the preconditioner of a matrix using AMG.
     */
    PETScAMGPreconditioner(
      const std::string   &name                             = "",
      const bool           symmetric_operator               = false,
      const double         strong_threshold                 = 0.25,
      const double         max_row_sum                      = 0.9,
      const unsigned int   aggressive_coarsening_num_levels = 0,
      const bool           output_details                   = false,
      const RelaxationType relaxation_type_up   = RelaxationType::SORJacobi,
      const RelaxationType relaxation_type_down = RelaxationType::SORJacobi,
      const RelaxationType relaxation_type_coarse =
        RelaxationType::GaussianElimination,
      const unsigned int n_sweeps_coarse = 1,
      const double       tol             = 0.0,
      const unsigned int max_iter        = 1,
      const bool         w_cycle         = false);

    /**
     * Initialize the preconditioner using @p matrix.
     */
    void
    initialize(const dealii::PETScWrappers::MatrixBase &matrix);

  private:
    /**
     * Declare preconditioner options.
     */
    void
    add_parameters();

    /**
     * Set this flag to true if you have a symmetric system matrix and you
     * want to use a solver which assumes a symmetric preconditioner like
     * CG.
     */
    bool symmetric_operator;

    /**
     * Threshold of when nodes are considered strongly connected. See
     * HYPRE_BoomerAMGSetStrongThreshold(). Recommended values are 0.25 for
     * 2d and 0.5 for 3d problems, but it is problem dependent.
     */
    double strong_threshold;

    /**
     * If set to a value smaller than 1.0 then diagonally dominant parts of
     * the matrix are treated as having no strongly connected nodes. If the
     * row sum weighted by the diagonal entry is bigger than the given
     * value, it is considered diagonally dominant. This feature is turned
     * of by setting the value to 1.0. This is the default as some matrices
     * can result in having only diagonally dominant entries and thus no
     * multigrid levels are constructed. The default in BoomerAMG for this
     * is 0.9. When you try this, check for a reasonable number of levels
     * created.
     */
    double max_row_sum;

    /**
     * Number of levels of aggressive coarsening. Increasing this value
     * reduces the construction time and memory requirements but may
     * decrease effectiveness.
     */
    unsigned int aggressive_coarsening_num_levels;

    /**
     * Setting this flag to true produces debug output from HYPRE, when the
     * preconditioner is constructed.
     */
    bool output_details;

    /**
     * Choose relaxation type up.
     */
    RelaxationType relaxation_type_up;

    /**
     * Choose relaxation type down.
     */
    RelaxationType relaxation_type_down;

    /**
     * Choose relaxation type coarse.
     */
    RelaxationType relaxation_type_coarse;

    /**
     * Choose number of sweeps on coarse grid.
     */
    unsigned int n_sweeps_coarse;

    /**
     * Choose BommerAMG tolerance.
     */
    double tol;

    /**
     * Choose BommerAMG maximum number of cycles.
     */
    unsigned int max_iter;

    /**
     * Defines whether a w-cycle should be used instead of the standard
     * setting of a v-cycle.
     */
    bool w_cycle;
  };


} // namespace ParsedLAC

#endif // DEAL_II_WITH_PETSC

#endif
