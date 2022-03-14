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
#ifndef pdes_mpi_linear_elasticity_h
#define pdes_mpi_linear_elasticity_h

#include "parsed_tools/constants.h"
#include "pdes/linear_problem.h"

namespace PDEs
{
  using namespace dealii;

  /**
   * Solve the LinearElasticity problem.
   */
  template <int dim, int spacedim = dim, class LacType = LAC::LAdealii>
  class LinearElasticity : public LinearProblem<dim, spacedim, LacType>
  {
  public:
    /**
     * Constructor. Initialize all parameters, including the base class, and
     * make sure the class is ready to run.
     */
    LinearElasticity();

    /**
     * Destroy the LinearElasticity object
     */
    virtual ~LinearElasticity() = default;

    using ScratchData =
      typename LinearProblem<dim, spacedim, LacType>::ScratchData;

    using CopyData = typename LinearProblem<dim, spacedim, LacType>::CopyData;

    using VectorType =
      typename LinearProblem<dim, spacedim, LacType>::VectorType;

  protected:
    /**
     * Explicitly assemble the LinearElasticity problem on a single cell.
     */
    virtual void
    assemble_system_one_cell(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      ScratchData &                                                   scratch,
      CopyData &copy) override;

    /**
     * Make sure we initialize the right type of linear solver.
     */
    virtual void
    solve() override;

    ParsedTools::Function<spacedim>  lambda;
    ParsedTools::Function<spacedim>  mu;
    const FEValuesExtractors::Vector displacement;
  };

  namespace MPI
  {
    template <int dim, int spacedim = dim>
    using LinearElasticity =
      PDEs::LinearElasticity<dim, spacedim, LAC::LATrilinos>;
  }

  namespace Serial
  {
    template <int dim, int spacedim = dim>
    using LinearElasticity =
      PDEs::LinearElasticity<dim, spacedim, LAC::LAdealii>;
  }
} // namespace PDEs
#endif