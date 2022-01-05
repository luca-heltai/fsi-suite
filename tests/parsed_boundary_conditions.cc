#include <deal.II/base/config.h>

#include "tools/parsed_boundary_conditions.h"

#include <gtest/gtest.h>

#include "dim_spacedim_tester.h"

#ifdef DEAL_II_WITH_SYMENGINE
#  include <fstream>
#  include <sstream>

using namespace dealii;

TYPED_TEST(DimTester, ParsedBoundaryConditions)
{
  Tools::ParsedBoundaryConditions<this->dim> pbc("/");

  this->parse(
    "set ids:components:bc_type:expressions (u) = -1:u:dirichlet:x^2");
}
#endif