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

#ifndef fsi_lac_h
#define fsi_lac_h

#include <deal.II/base/config.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>

/**
 * Wrappers for linear algebra classes. This collection of structs allows us to
 * write codes that are independent on the linear algebra backend used.
 *
 * The first class using this collection is the PDEs::LinearProblem class, that
 * represents an abstract matrix based linear problem, and provides all boiler
 * plate codes required for the assembly, solution, and post-processing of a
 * general (multicomponent) linear problem, such as Poisson, LinearElasticity,
 * or Stokes problem.
 */
namespace LAC
{
  /**
   * Serial linear algebra, natively defined in deal.II.
   *
   * We use this class to specify block vectors, matrices, and sparsity patterns
   * to use with serial Triangulation objects.
   */
  struct LAdealii
  {
    using BlockVector          = dealii::BlockVector<double>;
    using BlockSparseMatrix    = dealii::BlockSparseMatrix<double>;
    using BlockSparsityPattern = dealii::BlockSparsityPattern;
  };


#ifdef DEAL_II_WITH_PETSC

#  include <deal.II/lac/petsc_block_sparse_matrix.h>
#  include <deal.II/lac/petsc_solver.h>

  /**
   * Parallel linear algebra, using PETSc.
   *
   * We use this class to specify block vectors, matrices, and sparsity patterns
   * to use with parallel Triangulation objects.
   */
  struct LAPETSc
  {
    using BlockVector          = dealii::PETScWrappers::MPI::BlockVector;
    using BlockSparseMatrix    = dealii::PETScWrappers::MPI::BlockSparseMatrix;
    using BlockSparsityPattern = dealii::BlockSparsityPattern;
  };

#endif // DEAL_II_WITH_PETSC

#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/lac/block_sparsity_pattern.h>
#  include <deal.II/lac/trilinos_block_sparse_matrix.h>
#  include <deal.II/lac/trilinos_vector.h>

  /**
   * Parallel linear algebra, using Trilinos.
   *
   * We use this class to specify block vectors, matrices, and sparsity patterns
   * to use with parallel Triangulation objects.
   */
  struct LATrilinos
  {
    using BlockVector          = dealii::TrilinosWrappers::MPI::BlockVector;
    using BlockSparseMatrix    = dealii::TrilinosWrappers::BlockSparseMatrix;
    using BlockSparsityPattern = dealii::TrilinosWrappers::BlockSparsityPattern;
  };

#endif // DEAL_II_WITH_TRILINOS
} // namespace LAC
#endif