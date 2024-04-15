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
#ifndef pdes_stokes_h
#define pdes_stokes_h

#include "parsed_lac/amg.h"
#include "parsed_lac/ilu.h"
#include "parsed_lac/inverse_operator.h"
#include "parsed_tools/constants.h"
#include "pdes/linear_problem.h"

namespace PDEs
{
  using namespace dealii;

  /**
   * Solve the Stokes problem, in parallel.
   */
  template <int dim, class LacType>
  class Stokes : public LinearProblem<dim, dim, LacType>
  {
  public:
    /**
     * Constructor. Initialize all parameters, including the base class, and
     * make sure the class is ready to run.
     */
    Stokes();

    /**
     * Destroy the Stokes object
     */
    virtual ~Stokes() = default;

    using ScratchData = typename LinearProblem<dim, dim, LacType>::ScratchData;

    using CopyData = typename LinearProblem<dim, dim, LacType>::CopyData;

    using VectorType = typename LinearProblem<dim, dim, LacType>::VectorType;

  protected:
    /**
     * Explicitly assemble the Stokes problem on a single cell.
     */
    virtual void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData                                          &scratch,
      CopyData                                             &copy) override;

    /**
     * Make sure we initialize the right type of linear solver.
     */
    virtual void
    solve() override;

    ParsedTools::Constants     constants;
    typename LacType::AMG      schur_preconditioner;
    ParsedLAC::InverseOperator schur_solver;


    const FEValuesExtractors::Vector velocity;
    const FEValuesExtractors::Scalar pressure;
  };

  namespace MPI
  {
    template <int dim>
    using Stokes = PDEs::Stokes<dim, LAC::LATrilinos>;
  }

  namespace Serial
  {
    template <int dim>
    using Stokes = PDEs::Stokes<dim, LAC::LAdealii>;
  }
} // namespace PDEs
#endif