#include "tools/parsed_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimTester, ParsedFunctionConstruction)
{
  Tools::ParsedFunction<this->dim> function(this->id("fun"), "x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 1.0);
}


TYPED_TEST(DimTester, ParsedFunctionParsing)
{
  Tools::ParsedFunction<this->dim> function(this->id("fun"), "x");

  this->parse(R"(
    set Function expression = 2*x
  )",
              function);

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p), 2.0);
}



TYPED_TEST(DimTester, ParsedFunctionParsingVectorValued)
{
  Tools::ParsedFunction<this->dim> function(this->id("fun"), "x; 2*x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function.value(p, 0), 1.0);
  ASSERT_EQ(function.value(p, 1), 2.0);

  this->parse(R"(
    set Function expression = 2*x; 3*x
  )",
              function);

  ASSERT_EQ(function.value(p, 0), 2.0);
  ASSERT_EQ(function.value(p, 1), 3.0);

  // Make sure that the function throws if we pass the wrong number of
  // expressions
  ASSERT_ANY_THROW({
    this->parse(R"(
        set Function expression = x
      )",
                function);
  });
}