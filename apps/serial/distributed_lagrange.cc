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

#include "pdes/distributed_lagrange.h"

#include "runner.h"

/**
 * Parallel distributed Lagrange finite element method.
 */
int
main(int argc, char **argv)
{
  RUNNER_CODIM(PDEs::MPI::DistributedLagrange, argc, argv);
}
