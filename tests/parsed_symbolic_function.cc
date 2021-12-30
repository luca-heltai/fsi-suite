#include "tools/parsed_symbolic_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

TYPED_TEST(DT, ParsedSymbolicFunctionConstruction)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/", "x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 1.0);
}


TYPED_TEST(DT, ParsedSymbolicFunctionParsing)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/", "x");

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Function expression = 2*x
  )");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 2.0);
}



TYPED_TEST(DT, ParsedSymbolicFunctionParsingVectorValued)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/", "x; 2*x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p, 0), 1.0);
  ASSERT_EQ(function().value(p, 1), 2.0);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Function expression = 2*x; 3*x
  )");

  ASSERT_EQ(function().value(p, 0), 2.0);
  ASSERT_EQ(function().value(p, 1), 3.0);

  function().print(std::cout);
}
#endif