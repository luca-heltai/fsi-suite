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

namespace LAC
{
  /**
   * General class, used to initialize different types of block vectors, block
   * atrices and block sparsity patterns using a common interface.
   */
  class BlockInitializer
  {
  public:
    BlockInitializer(
      const std::vector<dealii::types::global_dof_index> &dofs_per_block,
      const std::vector<dealii::IndexSet>                &owned,
      const std::vector<dealii::IndexSet>                &relevant,
      const MPI_Comm                                     &comm = MPI_COMM_WORLD)
      : dofs_per_block(dofs_per_block)
      , owned(owned)
      , relevant(relevant)
      , comm(comm){};

    /**
     * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
     */
    void
    operator()(LATrilinos::BlockVector &v, bool fast = false)
    {
      v.reinit(owned, comm, fast);
    };


    /**
     * Initialize a ghosted TrilinosWrappers::MPI::BlockVector.
     */
    void
    ghosted(LATrilinos::BlockVector &v, bool fast = false)
    {
      v.reinit(owned, relevant, comm, fast);
    };

    /**
     * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
     */
    void
    operator()(LAPETSc::BlockVector &v, bool fast = false)
    {
      (void)fast;
      v.reinit(owned, comm);
    };


    /**
     * Initialize a ghosted TrilinosWrappers::MPI::BlockVector.
     */
    void
    ghosted(LAPETSc::BlockVector &v, bool fast = false)
    {
      (void)fast;
      v.reinit(owned, relevant, comm);
    };

    /**
     * Initialize a serial BlockVector<double>.
     */
    void
    operator()(LAdealii::BlockVector &v, bool fast = false)
    {
      v.reinit(dofs_per_block, fast);
    };


    /**
     * Initiale a ghosted BlockVector<double>. Same as above.
     */
    void
    ghosted(LAdealii::BlockVector &v, bool fast = false)
    {
      v.reinit(dofs_per_block, fast);
    };

    /**
     * Initialize a Trilinos Sparsity Pattern.
     */
    template <int dim, int spacedim>
    void
    operator()(dealii::TrilinosWrappers::BlockSparsityPattern     &s,
               const dealii::DoFHandler<dim, spacedim>            &dh,
               const dealii::AffineConstraints<double>            &cm,
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
    operator()(dealii::BlockSparsityPattern                       &s,
               const dealii::DoFHandler<dim, spacedim>            &dh,
               const dealii::AffineConstraints<double>            &cm,
               const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
    {
      dsp =
        std::make_unique<dealii::BlockDynamicSparsityPattern>(dofs_per_block,
                                                              dofs_per_block);

      dealii::DoFTools::make_sparsity_pattern(dh, coupling, *dsp, cm, false);
      dsp->compress();
      s.copy_from(*dsp);
    }

    /**
     * Initialize a deal.II matrix
     */
    void
    operator()(const LAdealii::BlockSparsityPattern &sparsity,
               LAdealii::BlockSparseMatrix          &matrix)
    {
      matrix.reinit(sparsity);
    };


    /**
     * Initialize a Trilinos matrix
     */
    void
    operator()(const LATrilinos::BlockSparsityPattern &sparsity,
               LATrilinos::BlockSparseMatrix          &matrix)
    {
      matrix.reinit(sparsity);
    };



    /**
     * Initialize a PETSc matrix
     */
    void
    operator()(const LAPETSc::BlockSparsityPattern &,
               LAPETSc::BlockSparseMatrix &matrix)
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

  /**
   * General class, used to initialize different types of block vectors, block
   * atrices and block sparsity patterns using a common interface.
   */
  class Initializer
  {
  public:
    Initializer(const dealii::IndexSet &owned_rows,
                const dealii::IndexSet &relevant_rows,
                const MPI_Comm         &comm             = MPI_COMM_WORLD,
                const dealii::IndexSet &owned_columns    = dealii::IndexSet(),
                const dealii::IndexSet &relevant_columns = dealii::IndexSet())
      : owned_rows(owned_rows)
      , relevant_rows(relevant_rows)
      , comm(comm)
      , owned_columns(owned_columns.size() > 0 ? owned_columns : owned_rows)
      , relevant_columns(relevant_columns.size() > 0 ? relevant_columns :
                                                       relevant_rows){};

    /**
     * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
     */
    void
    operator()(LATrilinos::Vector &v, bool fast = false)
    {
      v.reinit(owned_rows, comm, fast);
    };


    /**
     * Initialize a ghosted TrilinosWrappers::MPI::Vector.
     */
    void
    ghosted(LATrilinos::Vector &v, bool fast = false)
    {
      v.reinit(owned_rows, relevant_rows, comm, fast);
    };

    /**
     * Initialize a non ghosted TrilinosWrappers::MPI::Vector.
     */
    void
    operator()(LAPETSc::Vector &v, bool fast = false)
    {
      (void)fast;
      v.reinit(owned_rows, comm);
    };


    /**
     * Initialize a ghosted TrilinosWrappers::MPI::Vector.
     */
    void
    ghosted(LAPETSc::Vector &v, bool fast = false)
    {
      (void)fast;
      v.reinit(owned_rows, relevant_rows, comm);
    };

    /**
     * Initialize a serial BlockVector<double>.
     */
    void
    operator()(LAdealii::Vector &v, bool fast = false)
    {
      v.reinit(owned_rows.size(), fast);
    };


    /**
     * Initiale a ghosted BlockVector<double>. Same as above.
     */
    void
    ghosted(LAdealii::Vector &v, bool fast = false)
    {
      v.reinit(owned_rows.size(), fast);
    };

    /**
     * Initialize a Trilinos Sparsity Pattern.
     */
    void
    operator()(dealii::TrilinosWrappers::SparsityPattern &s)
    {
      s.reinit(owned_rows, owned_columns, comm);
    }

    /**
     * Initialize a deal.II Sparsity Pattern.
     */
    void
    operator()(dealii::SparsityPattern &s)
    {
      s.reinit(owned_rows.size(), owned_columns.size(), 0);
    }

    /**
     * Initialize a Trilinos Sparsity Pattern.
     */
    template <int dim, int spacedim>
    void
    operator()(dealii::TrilinosWrappers::SparsityPattern          &s,
               const dealii::DoFHandler<dim, spacedim>            &dh,
               const dealii::AffineConstraints<double>            &cm,
               const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
    {
      s.reinit(owned_rows, owned_columns, relevant_rows, comm);
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
    operator()(dealii::SparsityPattern                            &s,
               const dealii::DoFHandler<dim, spacedim>            &dh,
               const dealii::AffineConstraints<double>            &cm,
               const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
    {
      dsp =
        std::make_unique<dealii::DynamicSparsityPattern>(owned_rows.size(),
                                                         owned_columns.size());

      dealii::DoFTools::make_sparsity_pattern(dh, coupling, *dsp, cm, false);
      dsp->compress();
      s.copy_from(*dsp);
    }

    /**
     * Initialize a deal.II matrix
     */
    void
    operator()(const LAdealii::SparsityPattern &sparsity,
               LAdealii::SparseMatrix          &matrix)
    {
      matrix.reinit(sparsity);
    };


    /**
     * Initialize a Trilinos matrix
     */
    void
    operator()(const LATrilinos::SparsityPattern &sparsity,
               LATrilinos::SparseMatrix          &matrix)
    {
      matrix.reinit(sparsity);
    };



    /**
     * Initialize a PETSc matrix
     */
    void
    operator()(const LAPETSc::SparsityPattern &sparsity,
               LAPETSc::SparseMatrix          &matrix)
    {
      matrix.reinit(owned_rows, owned_columns, sparsity, comm);
    };


  private:
    /**
     * The dynamic sparisty pattern.
     */
    std::unique_ptr<dealii::DynamicSparsityPattern> dsp;

    /**
     * Owned dofs.
     */
    const dealii::IndexSet owned_rows;
    const dealii::IndexSet relevant_rows;
    /**
     * MPI Communicator.
     */
    const MPI_Comm &comm;

    /**
     * Relevant dofs.
     */
    const dealii::IndexSet owned_columns;
    const dealii::IndexSet relevant_columns;
  };
} // namespace LAC
#endif
