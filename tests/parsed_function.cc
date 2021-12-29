#include "tools/parsed_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DT, ParsedFunctionConstruction)
{
  Tools::ParsedFunction<this->dim> function("/", "x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 1.0);
}


TYPED_TEST(DT, ParsedFunctionParsing)
{
  Tools::ParsedFunction<this->dim> function("/", "x");

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Function expression = 2*x
  )");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 2.0);
}



TYPED_TEST(DT, ParsedFunctionParsingVectorValued)
{
  Tools::ParsedFunction<this->dim> function("/", "x; 2*x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p, 0), 1.0);
  ASSERT_EQ(function.value(p, 1), 2.0);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Function expression = 2*x; 3*x
  )");

  ASSERT_EQ(function.value(p, 0), 2.0);
  ASSERT_EQ(function.value(p, 1), 3.0);

  // Make sure that the function throws if we pass the wrong number of
  // expressions

  try
    {
      ParameterAcceptor::prm.parse_input_from_string(R"(
        set Function expression = x
      )");
      FAIL() << "Expected an exception";
    }
  catch (...)
    {
      SUCCEED();
    }
}