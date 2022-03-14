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

#ifndef fsi_lac_initializer_h
#define fsi_lac_initializer_h

// This includes all types we know of.
#include <mpi.h>

#include "lac.h"

/**
 * General class, used to initialize different types of Vectors, Matrices and
 * Sparsity Patterns using a common interface.
 */
class ScopedLACInitializer
{
public:
  ScopedLACInitializer(
    const std::vector<dealii::types::global_dof_index> &dofs_per_block,
    const std::vector<dealii::IndexSet> &               owned,
    const std::vector<dealii::IndexSet> &               relevant,
    const MPI_Comm &                                    comm = MPI_COMM_WORLD)
    : dofs_per_block(dofs_per_block)
    , owned(owned)
    , relevant(relevant)
    , comm(comm){};

  /**
   * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void
  operator()(dealii::TrilinosWrappers::MPI::BlockVector &v, bool fast = false)
  {
    v.reinit(owned, comm, fast);
  };


  /**
   * Initialize a ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void
  ghosted(dealii::TrilinosWrappers::MPI::BlockVector &v, bool fast = false)
  {
    v.reinit(owned, relevant, comm, fast);
  };

  /**
   * Initialize a serial BlockVector<double>.
   */
  void
  operator()(dealii::BlockVector<double> &v, bool fast = false)
  {
    v.reinit(dofs_per_block, fast);
  };


  /**
   * Initiale a ghosted BlockVector<double>. Same as above.
   */
  void
  ghosted(dealii::BlockVector<double> &v, bool fast = false)
  {
    v.reinit(dofs_per_block, fast);
  };

  /**
   * Initialize a Trilinos Sparsity Pattern.
   */
  template <int dim, int spacedim>
  void
  operator()(dealii::TrilinosWrappers::BlockSparsityPattern &    s,
             const dealii::DoFHandler<dim, spacedim> &           dh,
             const dealii::AffineConstraints<double> &           cm,
             const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
  {
    s.reinit(owned, owned, relevant, comm);
    dealii::DoFTools::make_sparsity_pattern(
      dh,
      coupling,
      s,
      cm,
      false,
      dealii::Utilities::MPI::this_mpi_process(comm));
    s.compress();
  }

  /**
   * Initialize a Deal.II Sparsity Pattern.
   */
  template <int dim, int spacedim>
  void
  operator()(dealii::BlockSparsityPattern &                      s,
             const dealii::DoFHandler<dim, spacedim> &           dh,
             const dealii::AffineConstraints<double> &           cm,
             const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
  {
    dsp = std::make_unique<dealii::BlockDynamicSparsityPattern>(dofs_per_block,
                                                                dofs_per_block);

    dealii::DoFTools::make_sparsity_pattern(dh, coupling, *dsp, cm, false);
    dsp->compress();
    s.copy_from(*dsp);
  }

  /**
   * Initialize a deal.II matrix
   */
  void
  operator()(const LAC::LAdealii::BlockSparsityPattern &sparsity,
             LAC::LAdealii::BlockSparseMatrix &         matrix)
  {
    matrix.reinit(sparsity);
  };


  /**
   * Initialize a Trilinos matrix
   */
  void
  operator()(const LAC::LATrilinos::BlockSparsityPattern &sparsity,
             LAC::LATrilinos::BlockSparseMatrix &         matrix)
  {
    matrix.reinit(sparsity);
  };



  /**
   * Initialize a PETSc matrix
   */
  void
  operator()(const LAC::LAPETSc::BlockSparsityPattern &,
             LAC::LAPETSc::BlockSparseMatrix &matrix)
  {
    Assert(dsp, dealii::ExcNotInitialized());
    matrix.reinit(owned, *dsp, comm);
  };


private:
  /**
   * The dynamic sparisty pattern.
   */
  std::unique_ptr<dealii::BlockDynamicSparsityPattern> dsp;

  /**
   * Dofs per block.
   */
  const std::vector<dealii::types::global_dof_index> &dofs_per_block;

  /**
   * Owned dofs per block.
   */
  const std::vector<dealii::IndexSet> &owned;

  /**
   * Relevant dofs per block.
   */
  const std::vector<dealii::IndexSet> &relevant;

  /**
   * MPI Communicator.
   */
  const MPI_Comm &comm;
};
#endif
