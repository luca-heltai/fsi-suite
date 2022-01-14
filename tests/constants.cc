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