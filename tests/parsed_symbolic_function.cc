#include "tools/parsed_symbolic_function.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "tools/parsed_constants.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

TYPED_TEST(DimTester, ParsedSymbolicFunctionConstruction)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/sf" + this->id(), "x");

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 1.0);
}


TYPED_TEST(DimTester, ParsedSymbolicFunctionParsing)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/sf2" + this->id(), "x");

  this->parse("set Function expression = 2*x", function);

  Point<this->dim> p;
  p[0] = 1.0;

  ASSERT_EQ(function().value(p), 2.0);
}



TYPED_TEST(DimTester, ParsedSymbolicFunctionAndConstants)
{
  Tools::ParsedConstants constants("/sc3" + this->id(), {"a"}, {2.0});

  Tools::ParsedSymbolicFunction<this->dim> function("/sc3" + this->id(), "a*x");

  Point<this->dim> p;
  p[0] = 1.0;
  function().update_user_substitution_map(constants);

  ASSERT_EQ(function().value(p), 2.0);

  this->parse("set a = 3", function);

  function().update_user_substitution_map(constants);
  ASSERT_EQ(function().value(p), 3.0);
}


TYPED_TEST(DimTester, ParsedSymbolicFunctionParsingVectorValued)
{
  Tools::ParsedSymbolicFunction<this->dim> function("/sf1" + this->id(),
                                                    "x; 2*x");

  Point<this->dim> p;
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