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
#ifndef pdes_mpi_poisson_h
#define pdes_mpi_poisson_h

#include "pdes/linear_problem.h"

namespace PDEs
{
  namespace MPI
  {
    using namespace dealii;

    /**
     * Solve the Poisson problem, in parallel.
     */
    template <int dim, int spacedim = dim>
    class Poisson : public LinearProblem<dim, spacedim, LAC::LATrilinos>
    {
    public:
      /**
       * Constructor. Initialize all parameters, including the base class, and
       * make sure the class is ready to run.
       */
      Poisson();

      /**
       * Destroy the Poisson object
       */
      virtual ~Poisson() = default;

      /**
       * Build a custom error estimator.
       */
      virtual void
      custom_estimator(dealii::Vector<float> &error_per_cell) const override;

      using ScratchData =
        typename LinearProblem<dim, spacedim, LAC::LATrilinos>::ScratchData;

      using CopyData =
        typename LinearProblem<dim, spacedim, LAC::LATrilinos>::CopyData;

      using VectorType =
        typename LinearProblem<dim, spacedim, LAC::LATrilinos>::VectorType;

    protected:
      /**
       * Explicitly assemble the Poisson problem on a single cell.
       */
      virtual void
      assemble_system_one_cell(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        ScratchData                                                    &scratch,
        CopyData &copy) override;

      /**
       * Make sure we initialize the right type of linear solver.
       */
      virtual void
      solve() override;

      ParsedTools::Function<spacedim> coefficient;
    };
  } // namespace MPI
} // namespace PDEs
#endif