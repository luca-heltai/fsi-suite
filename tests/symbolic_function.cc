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

#include "parsed_tools/symbolic_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "parsed_tools/constants.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

TYPED_TEST(DimTester, SymbolicFunctionConstruction)
{
  ParsedTools::SymbolicFunction<TestFixture::dim> function("/sf" + this->id(),
                                                           "x");

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 1.0);
}


TYPED_TEST(DimTester, SymbolicFunctionParsing)
{
  ParsedTools::SymbolicFunction<TestFixture::dim> function("/sf2" + this->id(),
                                                           "x");

  this->parse("set Function expression = 2*x", function);

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 2.0);
}



TYPED_TEST(DimTester, SymbolicFunctionAndConstants)
{
  ParsedTools::Constants constants("/sc3" + this->id(), {"a"}, {2.0});

  ParsedTools::SymbolicFunction<TestFixture::dim> function("/sc3" + this->id(),
                                                           "a*x");

  Point<TestFixture::dim> p;
  p[0] = 1.0;
  function().update_user_substitution_map(constants);

  ASSERT_EQ(function().value(p), 2.0);

  this->parse("set a = 3", function);

  function().update_user_substitution_map(constants);
  ASSERT_EQ(function().value(p), 3.0);
}


TYPED_TEST(DimTester, SymbolicFunctionParsingVectorValued)
{
  ParsedTools::SymbolicFunction<TestFixture::dim> function("/sf1" + this->id(),
                                                           "x; 2*x");

  Point<TestFixture::dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p, 0), 1.0);
  ASSERT_EQ(function().value(p, 1), 2.0);

  this->parse("set Function expression = 2*x; 3*x", function);

  ASSERT_EQ(function().value(p, 0), 2.0);
  ASSERT_EQ(function().value(p, 1), 3.0);

  // Now check that the function throws when the number of components is not
  // what the function has been built with
  ASSERT_ANY_THROW({ this->parse("set Function expression = 2*x", function); });
}
#endif