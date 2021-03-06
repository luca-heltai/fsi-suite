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

#include "parsed_tools/grid_refinement.h"

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTesterNoOne, GridRefinementGlobal)
{
  Triangulation<TestFixture::dim, TestFixture::spacedim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  ParsedTools::GridRefinement pgr("/");
  ParameterAcceptor::initialize();

  Vector<float> criteria(tria.n_active_cells());
  criteria = 1.0;
  pgr.mark_cells(criteria, tria);
  tria.execute_coarsening_and_refinement();

  ASSERT_EQ(tria.n_active_cells(), std::pow(2, (TestFixture::dim * 2)));
}