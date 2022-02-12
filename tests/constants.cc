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

#include "parsed_tools/constants.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "parsed_tools/function.h"

using namespace dealii;

TEST(Constants, CheckConstants)
{
  ParsedTools::Constants constants("/",
                                   {"a", "b", "c"},
                                   {1.0, 2.0, 3.0},
                                   {"The constant a",
                                    "The constant b",
                                    "The constant c"});

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set The constant a (a) = 4
    set The constant b (b) = 5
    set The constant c (c) = 6
  )");

  ASSERT_EQ(constants["a"], 4.0);
  ASSERT_EQ(constants["b"], 5.0);
  ASSERT_EQ(constants["c"], 6.0);
}


TEST(Constants, FunctionAndConstants)
{
  ParsedTools::Constants   constants("/",
                                   {"a", "b", "c"},
                                   {1.0, 2.0, 3.0},
                                   {"The constant a",
                                    "The constant b",
                                    "The constant c"});
  ParsedTools::Function<1> function(
    "/", "a*x^2+b*x+c", "Function expression", constants, "x,y");

  Point<1> p(1);

  ASSERT_EQ(function.value(p), 6.0);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set The constant a (a) = 1
    set The constant b (b) = 2
    set The constant c (c) = 0
  )");

  function.update_constants(constants);

  ASSERT_EQ(constants["c"], 0.0);
  ASSERT_EQ(function.value(p), 3.0);
}