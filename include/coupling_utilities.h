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

#ifndef fsi_suite_coupling_utilities_h
#define fsi_suite_coupling_utilities_h

#include <deal.II/base/config.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_vector.h>

namespace dealii
{
  namespace CouplingUtilities
  {
    /**
     * Join two affine constraints into a larger one that contains both, with
     * the second shifted.
     */
    template <typename NumberType>
    void
    join_constraints(const AffineConstraints<NumberType> &c1,
                     const AffineConstraints<NumberType> &c2,
                     AffineConstraints<NumberType>       &c1_c2);


    /**
     * Extract the given blocks from the full vector passed as second argument.
     * No copy is performed. The @p out_vec is a view of the @p in_vec.
     */
    template <int rows>
    void
    make_view(const std::array<unsigned int, rows> &indices,
              PETScWrappers::MPI::BlockVector      &in_vec,
              PETScWrappers::MPI::BlockVector      &out_vec);

    /**
     * Extract the given blocks from the full matrix. No copy is performed.
     * The @p out_mat is a view of the @p in_mat.
     */
    template <int rows, int cols>
    void
    make_view(const std::array<unsigned int, rows>  &row_indices,
              const std::array<unsigned int, cols>  &col_indices,
              PETScWrappers::MPI::BlockSparseMatrix &in_mat,
              PETScWrappers::MPI::BlockSparseMatrix &out_mat);
  } // namespace CouplingUtilities
} // namespace dealii

#ifndef DOXYGEN
// -------------------------------------------------------------------------
// ------------------------- IMPLEMENTATION -------------------------------
// -------------------------------------------------------------------------

namespace dealii
{
  namespace CouplingUtilities
  {
    template <typename NumberType>
    void
    join_constraints(const AffineConstraints<NumberType> &c1,
                     const AffineConstraints<NumberType> &c2,
                     AffineConstraints<NumberType>       &c1_c2)
    {
      const auto id1 = c1.get_local_lines();
      const auto id2 = c2.get_local_lines();

      Assert((id1.size() > 0 && id2.size() > 0),
             ExcMessage("Both constraints must specify local lines"));

      // Global index set
      IndexSet is(id1.size() + id2.size());
      is.add_indices(id1);
      is.add_indices(id2, id1.size());
      c1_c2.reinit(is);

      // Make a temporary copy of the second constraints, that we shift
      AffineConstraints<NumberType> c2tmp(c2);
      c2tmp.shift(id1.size());

      c1_c2.merge(c1,
                  AffineConstraints<NumberType>::no_conflicts_allowed,
                  true);
      c1_c2.merge(c2tmp,
                  AffineConstraints<NumberType>::no_conflicts_allowed,
                  true);
    }



    template <int rows>
    void
    make_view(const std::array<unsigned int, rows> &indices,
              PETScWrappers::MPI::BlockVector      &in_vec,
              PETScWrappers::MPI::BlockVector      &out_vec)
    {
      std::array<Vec, rows> vec_array;
      for (unsigned int i = 0; i < indices.size(); ++i)
        vec_array[i] = in_vec.block(indices[i]).petsc_vector();
      out_vec.reinit(PETScWrappers::MPI::BlockVector(vec_array).petsc_vector());
    }



    template <int rows, int cols>
    void
    make_view(const std::array<unsigned int, rows>  &row_indices,
              const std::array<unsigned int, cols>  &col_indices,
              PETScWrappers::MPI::BlockSparseMatrix &in_mat,
              PETScWrappers::MPI::BlockSparseMatrix &out_mat)
    {
      std::array<std::array<Mat, cols>, rows> mat_array;
      for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
          mat_array[i][j] =
            in_mat.block(row_indices[i], col_indices[j]).petsc_matrix();

      out_mat.reinit(
        PETScWrappers::MPI::BlockSparseMatrix(mat_array).petsc_matrix());
    }
  } // namespace CouplingUtilities
} // namespace dealii
#endif

#endif