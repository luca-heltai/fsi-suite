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
  Tools::ParsedBoundaryConditions<this->dim> pbc(this->id("pbc"));

  // Same as standard construction
  this->parse(R"(
       set Boundary id sets (u) = -1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
              pbc);

  // Check that we can parse more than one bc
  this->parse(R"(
       set Boundary id sets (u) = 1; 0,2
       set Selected components (u) = u; u
       set Boundary condition types (u) = dirichlet, dirichlet
       set Expressions (u) = 0 % 1
    )",
              pbc);

  // Expect failure due to mismatch between number of expressions and number of
  // ids
  ASSERT_ANY_THROW({
    this->parse(R"(
       set Boundary id sets (u) = 1,2; 3,4
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
                pbc);
  });

  // Check we fail if ids are set in the wrong way
  ASSERT_ANY_THROW({
    this->parse(R"(
       set Boundary id sets (u) = -1, 1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
                pbc);
  });

  // Check we fail if ids are set in the wrong way
  ASSERT_ANY_THROW({
    this->parse(R"(
       set Boundary id sets (u) = 1; 1
       set Selected components (u) = u; u
       set Boundary condition types (u) = dirichlet, dirichlet
       set Expressions (u) = 0 % 0
    )",
                pbc);
  });

  // Check that we fail if there are too many expressions
  ASSERT_ANY_THROW({
    this->parse(R"(
       set Boundary id sets (u) = 1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0; 1
    )",
                pbc);
  });
}
#endif