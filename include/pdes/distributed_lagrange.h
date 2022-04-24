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

#ifndef pdes_distributed_lagrange_h
#define pdes_distributed_lagrange_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

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
  template <int dim, int spacedim = dim, typename LacType = LAC::LAdealii>
  class DistributedLagrange : public dealii::ParameterAcceptor
  {
  public:
    DistributedLagrange();

    void
    run();

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

    PDEs::LinearProblem<spacedim, spacedim, LacType> space;
    GridTools::Cache<spacedim, spacedim>             space_cache;

    PDEs::LinearProblem<dim, spacedim, LacType> embedded;
    GridTools::Cache<dim, spacedim>             embedded_cache;

    ParsedTools::NonMatchingCoupling<dim, spacedim> coupling;

    typename LacType::SparsityPattern coupling_sparsity;
    typename LacType::SparseMatrix    coupling_matrix;

    ParsedLAC::InverseOperator mass_solver;
  };

  // namespace Serial
  // {
  //   template <int dim, int spacedim = dim>
  //   using DistributedLagrange =
  //     PDEs::DistributedLagrange<dim, spacedim, LAC::LAdealii>;
  // }

  namespace MPI
  {
    template <int dim, int spacedim = dim>
    using DistributedLagrange =
      PDEs::DistributedLagrange<dim, spacedim, LAC::LAPETSc>;
  }

  // namespace MPI
  // {
  //   template <int dim, int spacedim = dim>
  //   using DistributedLagrange =
  //     PDEs::DistributedLagrange<dim, spacedim, LAC::LATrilinos>;
  // }

} // namespace PDEs

#endif