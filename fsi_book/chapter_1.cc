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

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "pdes/mpi/poisson.h"

using namespace dealii;

TEST(Chapter1, Fig1_PoissonRegularGrid)
{
  static const int        dim = 2;
  PDEs::MPI::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_fig_01_poisson_regular_grid.prm",
    "chapter_1_fig_01_poisson_regular_grid.prm");
  poisson.run();
}