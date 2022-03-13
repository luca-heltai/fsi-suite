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

#include "parsed_tools/finite_element.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTester, FiniteElementScalar)
{
  ParsedTools::FiniteElement<TestFixture::dim, TestFixture::spacedim> pfe(
    this->id("pfe"), "u", "FE_Q(1)");

  this->parse(R"(
    set Finite element space (u) = FE_Q(1)
  )",
              pfe);

  FiniteElement<TestFixture::dim, TestFixture::spacedim> &fe = pfe;

  ASSERT_EQ(fe.n_components(), 1u);
  ASSERT_EQ(fe.n_blocks(), 1u);
  ASSERT_EQ(fe.n_base_elements(), 1u);
  ASSERT_EQ(fe.dofs_per_cell, std::pow(2, TestFixture::dim));
}