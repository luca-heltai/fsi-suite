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
#ifndef pdes_babuska_bc_h
#define pdes_babuska_bc_h

#include "pdes/coupled_problem.h"

namespace PDEs
{
  namespace MPI
  {
    using namespace dealii;

    /**
     * Solve the Poisson problem using a Lagrange multiplier to impose the
     * boundary conditions, in parallel.
     */
    template <int dim, int spacedim = dim>
    class BabuskaBC : public CoupledProblem<dim, dim - 1, spacedim>
    {
    public:
      /**
       * Constructor. Initialize all parameters, including the base class, and
       * make sure the class is ready to run.
       */
      BabuskaBC();

      /**
       * Destroy the BabuskaBC object
       */
      virtual ~BabuskaBC() = default;

    protected:
      ParsedTools::Function<spacedim> coefficient;

      ParsedLAC::InverseOperator schur_inverse_operator;
    };
  } // namespace MPI
} // namespace PDEs
#endif