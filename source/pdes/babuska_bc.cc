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

#include "pdes/babuska_bc.h"

#include <deal.II/meshworker/mesh_loop.h>

using namespace dealii;

namespace PDEs
{
  namespace MPI
  {
    template <int dim, int spacedim>
    BabuskaBC<dim, spacedim>::BabuskaBC()
      : CoupledProblem<dim, dim - 1, spacedim>("BabuskaBC",
                                               "u",
                                               "Poisson",
                                               "lambda",
                                               "Lagrange multiplier")
      , coefficient("/BabuskaBC/Functions", "1", "Diffusion coefficient")
      , schur_inverse_operator("/BabuskaBC/Solver/Schur",
                               "cg",
                               ParsedLAC::SolverControlType::iteration_number,
                               10,
                               1e-12,
                               1e-6)
    {}

    // Explicit instantiations
    template class BabuskaBC<2, 2>;
    template class BabuskaBC<2, 3>;
    template class BabuskaBC<3, 3>;

  } // namespace MPI
} // namespace PDEs