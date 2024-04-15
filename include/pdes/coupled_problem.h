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

#ifndef pdes_coupled_problem_h
#define pdes_coupled_problem_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_snes.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_ts.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "coupling_utilities.h"
#include "lac.h"
#include "parsed_lac/amg.h"
#include "parsed_lac/inverse_operator.h"
#include "parsed_tools/boundary_conditions.h"
#include "parsed_tools/constants.h"
#include "parsed_tools/data_out.h"
#include "parsed_tools/finite_element.h"
#include "parsed_tools/function.h"
#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_refinement.h"
#include "parsed_tools/mapping_eulerian.h"
#include "parsed_tools/non_matching_coupling.h"
#include "pdes/linear_problem.h"

using namespace dealii;
namespace PDEs
{
  template <int dim, int dim1 = dim, int spacedim = dim>
  class CoupledProblem : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Vector type.
     */
    using VectorType = PETScWrappers::MPI::Vector;

    /**
     * Block vector type.
     */
    using BlockVectorType = PETScWrappers::MPI::BlockVector;

    /**
     * Block sparse matrix type.
     */
    using BlockMatrixType = PETScWrappers::MPI::BlockSparseMatrix;

    /**
     * Sparse matrix type.
     */
    using MatrixType = PETScWrappers::MPI::SparseMatrix;

    CoupledProblem(const std::string &coupled_problem_name,
                   const std::string &primary_problem_variables,
                   const std::string &primary_problem_name,
                   const std::string &secondary_problem_variables,
                   const std::string &secondary_problem_name);

    void
    run();


    /**
     * This function is called after the constraints have been computed.
     */
    virtual void
    setup_sparsity(BlockDynamicSparsityPattern &dsp);

  private:
    void
    generate_grids();

    void
    setup_system();

    void
    assemble_system();

    void
    solve();

    void
    output_results(const unsigned int cycle);

    bool use_direct_solver;

    PDEs::LinearProblem<dim, spacedim, LAC::LAPETSc> primary_problem;
    GridTools::Cache<dim, spacedim>                  primary_problem_cache;

    PDEs::LinearProblem<dim1, spacedim, LAC::LAPETSc> secondary_problem;
    GridTools::Cache<dim1, spacedim>                  secondary_problem_cache;

    // Information about the coupling
    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<IndexSet>                locally_owned_dofs_per_block;
    std::vector<IndexSet>                locally_relevant_dofs_per_block;

    // Split by problem (2 blocks)
    std::vector<IndexSet> locally_owned_dofs_per_problem;
    std::vector<IndexSet> locally_relevant_dofs_per_problem;

    // Global index sets (single block)
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    mutable AffineConstraints<double> constraints;
    mutable AffineConstraints<double> hanging_nodes_constraints;
    mutable AffineConstraints<double> homogeneous_constraints;


    /**
     * The full system matrix. The diagonals come from the primary and secondary
     * problems, while the off-diagonal blocks are the coupling matrices.
     */
    BlockMatrixType system_matrix;

    /**
     * The two off-diagonal blocks of the system matrix.
     */
    BlockMatrixType primary_secondary_matrix;
    BlockMatrixType secondary_primary_matrix;

    /**
     * The full system solution.
     */
    BlockVectorType solution;
    BlockVectorType relevant_solution;

    BlockVectorType rhs;
  };

  // namespace Serial
  // {
  //   template <int dim, int spacedim = dim>
  //   using CoupledProblem =
  //     PDEs::CoupledProblem<dim, spacedim, LAC::LAdealii>;
  // }

  // namespace MPI
  // {
  //   template <int dim, int spacedim = dim>
  //   using CoupledProblem =
  //     PDEs::CoupledProblem<dim, spacedim, LAC::LAPETSc>;
  // }

  namespace MPI
  {
    template <int dim, int dim1, int spacedim = dim>
    using CoupledProblem = PDEs::CoupledProblem<dim, dim1, spacedim>;
  }

} // namespace PDEs

#endif