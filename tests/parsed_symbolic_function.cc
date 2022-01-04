#include "tools/parsed_symbolic_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "tools/parsed_constants.h"

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



TYPED_TEST(DT, ParsedSymbolicFunctionAndConstants)
{
  Tools::ParsedConstants constants("/", {"a"}, {2.0});

  Tools::ParsedSymbolicFunction<this->dim> function("/", "a*x");

  Point<this->dim> p;
  p[0] = 1.0;
  function().update_user_substitution_map(constants);

  ASSERT_EQ(function().value(p), 2.0);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set a = 3
  )");

  function().update_user_substitution_map(constants);
  ASSERT_EQ(function().value(p), 3.0);
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

  // Now check that the function throws when the number of components is not
  // what the function has been built with
  ASSERT_ANY_THROW({
    ParameterAcceptor::prm.parse_input_from_string(
      "set Function expression = 2*x");
  });
}
#endif