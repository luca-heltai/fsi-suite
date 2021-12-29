#include "tools/parsed_constants.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace dealii;

TEST(ParsedConstants, CheckConstants)
{
  Tools::ParsedConstants constants("/",
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