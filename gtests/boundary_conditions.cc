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

#include <deal.II/base/config.h>

#include "parsed_tools/boundary_conditions.h"

#include <gtest/gtest.h>

#include "dim_spacedim_tester.h"

#ifdef DEAL_II_WITH_SYMENGINE
#  include <fstream>
#  include <sstream>

using namespace dealii;

TYPED_TEST(DimTester, BoundaryConditions)
{
  ParsedTools::BoundaryConditions<TestFixture::dim> pbc(this->id("pbc"));

  // Same as standard construction
  parse(R"(
       set Boundary id sets (u) = -1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
        pbc);

  // Check that we can parse more than one bc
  parse(R"(
       set Boundary id sets (u) = 1; 0,2
       set Selected components (u) = u; u
       set Boundary condition types (u) = dirichlet, dirichlet
       set Expressions (u) = 0 % 1
    )",
        pbc);

  // Expect failure due to mismatch between number of expressions and number of
  // ids
  ASSERT_ANY_THROW({
    parse(R"(
       set Boundary id sets (u) = 1,2; 3,4
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
          pbc);
  });

  // Check we fail if ids are set in the wrong way
  ASSERT_ANY_THROW({
    parse(R"(
       set Boundary id sets (u) = -1, 1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0
    )",
          pbc);
  });

  // Check we fail if ids are set in the wrong way
  ASSERT_ANY_THROW({
    parse(R"(
       set Boundary id sets (u) = 1; 1
       set Selected components (u) = u; u
       set Boundary condition types (u) = dirichlet, dirichlet
       set Expressions (u) = 0 % 0
    )",
          pbc);
  });

  // Check that we fail if there are too many expressions
  ASSERT_ANY_THROW({
    parse(R"(
       set Boundary id sets (u) = 1
       set Selected components (u) = u
       set Boundary condition types (u) = dirichlet
       set Expressions (u) = 0; 1
    )",
          pbc);
  });
}



TEST_F(TwoTester, BoundaryConditionsVector)
{
  ParsedTools::BoundaryConditions<dim> pbc(
    "/",
    "u, u, p",
    {{dealii::numbers::internal_face_boundary_id}},
    {"u"},
    {ParsedTools::BoundaryConditionType::dirichlet},
    {"0; 0; 0"});
  ParameterAcceptor::initialize("", "test.prm");

  // Same as standard construction
  ASSERT_NO_THROW({
    parse(R"(
       set Boundary id sets (u, u, p) = -1
       set Selected components (u, u, p) = u
       set Boundary condition types (u, u, p) = dirichlet
       set Expressions (u, u, p) = 0; 0; 0
    )",
          pbc);
  });


  // Expect failure due to mismatch between number of expressions and
  // number of selected components
  ASSERT_ANY_THROW({
    parse(R"(
         set Boundary id sets (u, u, p) = -1
         set Selected components (u, u, p) = u
         set Boundary condition types (u, u, p) = dirichlet
         set Expressions (u, u, p) = 0; 0
      )",
          pbc);
  });


  // Expect success, when the number of components of the expression matches the
  // vector nature of the normal component due to mismatch between number
  // of expressions and number of selected components
  ASSERT_NO_THROW({
    parse(R"(
           # set Boundary id sets (u, u, p) = -1
           set Selected components (u, u, p) = u.n
           set Boundary condition types (u, u, p) = dirichlet
           set Expressions (u, u, p) = 0; 0
        )",
          pbc);
  });

  // Expect failure, when asking the normal component of a scalar variable.
  ASSERT_ANY_THROW({
    parse(R"(
         set Boundary id sets (u, u, p) = -1
         set Selected components (u, u, p) = p.n
         set Boundary condition types (u, u, p) = dirichlet
         set Expressions (u, u, p) = 0
      end
      )",
          pbc);
  });

  // Try a mixture of BC:
  ASSERT_NO_THROW({
    parse(R"(
         set Boundary id sets (u, u, p) = 0; 1; 2,3
         set Selected components (u, u, p) = u; u.t; p
         set Boundary condition types (u, u, p) = dirichlet, dirichlet, neumann
         set Expressions (u, u, p) = 0;0;0 % 0;0 % 0;0;0
      )",
          pbc);
  });
}
#endif