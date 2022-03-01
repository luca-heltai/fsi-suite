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
#include "pdes/mpi/linear_problem.h"

namespace PDEs
{
  namespace MPI
  {
    using namespace dealii;

    /**
     * Solve the LinearElasticity problem, in parallel.
     */
    template <int dim>
    class LinearElasticity : public LinearProblem<dim>
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

      using ScratchData = typename LinearProblem<dim>::ScratchData;
      using CopyData    = typename LinearProblem<dim>::CopyData;

    protected:
      /**
       * Explicitly assemble the LinearElasticity problem on a single cell.
       */
      virtual void
      assemble_system_one_cell(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        ScratchData &                                         scratch,
        CopyData &                                            copy) override;

      /**
       * Make sure we initialize the right type of linear solver.
       */
      virtual void
      solve() override;

      ParsedTools::Function<dim>       coefficient;
      ParsedTools::Constants           constants;
      const FEValuesExtractors::Vector displacement;
    };
  } // namespace MPI
} // namespace PDEs
#endif