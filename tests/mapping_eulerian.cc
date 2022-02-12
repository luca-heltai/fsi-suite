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

#include "parsed_tools/mapping_eulerian.h"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTester, MappingEulerian)
{
  Triangulation<this->dim, this->spacedim> tria;
  GridGenerator::hyper_cube(tria);
  FESystem<this->dim, this->spacedim> fe(FE_Q<this->dim, this->spacedim>(1),
                                         this->spacedim);

  DoFHandler<this->dim, this->spacedim> dh(tria);
  dh.distribute_dofs(fe);

  Vector<double> displacement(dh.n_dofs());

  ParsedTools::MappingEulerian<this->dim, this->spacedim> mff(dh,
                                                              this->id("mff"));

  ASSERT_NO_THROW({
    parse(R"(
      set Initial configuration or displacement = 
      set Use displacement = false
    )",
          mff);
  });

  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 1.);

  // Now the mapping should be the same as the displacement
  Point<this->dim> p;
  for (unsigned int i = 0; i < this->dim; ++i)
    p[i] = 0.5;

  // Try transforming the point to real space
  auto real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should coincide
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], p[i]);

  // Try the same, but using the displacement.
  ASSERT_NO_THROW({
    parse(R"(
      set Initial configuration or displacement = 
      set Use displacement = true
    )",
          mff);
  });

  // This should be the zero vector now
  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 0.);

  // Again, try transforming the point to real space
  real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should coincide
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], p[i]);

  std::string id_expression;
  {
    // The id expression
    const std::string id[3] = {"x", "y", "z"};
    std::string       sep   = "";
    for (unsigned int i = 0; i < this->spacedim; ++i)
      {
        id_expression += sep + id[i];
        sep = "; ";
      }
  }

  // This should be the same as the identity.
  ASSERT_NO_THROW({
    parse("set Initial configuration or displacement = " + id_expression +
            "\nset Use displacement = false",
          mff);
  });

  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 1.);

  // Again, try transforming the point to real space
  real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should coincide
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], p[i]);

  // Now try the same with a displacement field, i.e., the configuration is
  // equal to twice the original configuration
  ASSERT_NO_THROW({
    parse("set Initial configuration or displacement = " + id_expression +
            "\nset Use displacement = true",
          mff);
  });

  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 1.);

  // Try again transforming the point to real space
  real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should *not* coincide
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], 2 * p[i]);
}



TYPED_TEST(DimTesterNoOne, MappingEulerianSimplices)
{
  Triangulation<this->dim, this->spacedim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);
  FESystem<this->dim, this->spacedim> fe(
    FE_SimplexP<this->dim, this->spacedim>(1), this->spacedim);

  DoFHandler<this->dim, this->spacedim> dh(tria);
  dh.distribute_dofs(fe);

  Vector<double> displacement(dh.n_dofs());

  ParsedTools::MappingEulerian<this->dim, this->spacedim> mff(dh,
                                                              this->id("mff"));

  ASSERT_NO_THROW({
    parse(R"(
      set Initial configuration or displacement = 
      set Use displacement = false
    )",
          mff);
  });

  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 1.);

  // Now the mapping should be the same as the displacement
  Point<this->dim> p;
  for (unsigned int i = 0; i < this->dim; ++i)
    p[i] = 0.5;

  // Try transforming the point to real space
  auto real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should coincide, since the reference simplex is
  // the same as the first active cell
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], p[i]);

  // Try the same, but using the displacement.
  ASSERT_NO_THROW({
    parse(R"(
      set Initial configuration or displacement = 
      set Use displacement = true
    )",
          mff);
  });

  // This should throw, since with
  // simplices it should not work.
  ASSERT_ANY_THROW(mff.initialize(displacement));

  std::string id_expression;
  {
    // The id expression
    const std::string id[3] = {"x", "y", "z"};
    std::string       sep   = "";
    for (unsigned int i = 0; i < this->spacedim; ++i)
      {
        id_expression += sep + id[i];
        sep = "; ";
      }
  }

  // This should be the same as the identity.
  ASSERT_NO_THROW({
    parse("set Initial configuration or displacement = " + id_expression +
            "\nset Use displacement = false",
          mff);
  });

  mff.initialize(displacement);
  ASSERT_DOUBLE_EQ(displacement.linfty_norm(), 1.);

  // Again, try transforming the point to real space
  real_p = mff().transform_unit_to_real_cell(tria.begin_active(), p);

  // In this case, the points should coincide
  for (unsigned int i = 0; i < this->dim; ++i)
    ASSERT_DOUBLE_EQ(real_p[i], p[i]);

  // Now try the same with a displacement field, i.e., the configuration is
  // equal to twice the original configuration. Again, this should throw, since
  // simplices do not support displacement mappings.
  ASSERT_NO_THROW({
    parse("set Initial configuration or displacement = " + id_expression +
            "\nset Use displacement = true",
          mff);
  });
  ASSERT_ANY_THROW(mff.initialize(displacement));
}