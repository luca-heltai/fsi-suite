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

#include "parsed_tools/function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimTester, FunctionConstruction)
{
  ParsedTools::Function<TestFixture::dim> function(this->id("fun"), "x");

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 1.0);
}


TYPED_TEST(DimTester, FunctionParsing)
{
  ParsedTools::Function<TestFixture::dim> function(this->id("fun"), "x");

  this->parse(R"(
    set Function expression = 2*x
  )",
              function);

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 2.0);
}



TYPED_TEST(DimTester, FunctionParsingVectorValued)
{
  ParsedTools::Function<TestFixture::dim> function(this->id("fun"), "x; 2*x");

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p, 0), 1.0);
  ASSERT_EQ(function.value(p, 1), 2.0);

  this->parse(R"(
    set Function expression = 2*x; 3*x
  )",
              function);

  ASSERT_EQ(function.value(p, 0), 2.0);
  ASSERT_EQ(function.value(p, 1), 3.0);

  // // Make sure that the function throws if we pass the wrong number of
  // // expressions
  // ASSERT_ANY_THROW({
  //   this->parse(R"(
  //       set Function expression = x
  //     )",
  //               function);
  // });
}