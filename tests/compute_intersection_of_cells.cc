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

#include "compute_intersection_of_cells.h"

#include "dim_spacedim_tester.h"

using namespace dealii;

struct Test_function
{
  double
  operator()(const Point<2> &p)
  {
    return p[0] + p[1] - std::sin(p[0]);
  }
};

TEST(DimTester, Quadrature_Over_1D_Simple_Intersection)
{
  constexpr int dim0     = 1;
  constexpr int dim1     = 2;
  constexpr int spacedim = 2;

  Triangulation<dim0, spacedim> tria0;
  GridGenerator::hyper_cube<dim0, spacedim>(tria0, -1., +1.);
  FE_Nothing<dim0, spacedim> dummy_fe0;
  DoFHandler<dim0, spacedim> dh0(tria0);
  dh0.distribute_dofs(dummy_fe0);
  const auto &cell0 = dh0.begin_active();
  cell0->vertex(0)  = Point<spacedim>(
    -0.5, -0.9); // rotate the first grid a little bit just for testing purposes
  cell0->vertex(1) = Point<spacedim>(+0.6, 0.8);


  Triangulation<dim1, spacedim> tria1;
  GridGenerator::hyper_cube<spacedim>(tria1, -1., 1.);
  FE_Nothing<dim1, spacedim> dummy_fe1;
  DoFHandler<dim1, spacedim> dh1(tria1);
  dh1.distribute_dofs(dummy_fe1);
  const auto &cell1 = dh1.begin_active();



  const unsigned int degree = 4;
  auto               test_quadrature =
    compute_intersection<dim0, dim1, spacedim>(cell0, cell1, degree);


  const auto        &JxW       = test_quadrature.get_weights();
  const unsigned int quad_size = test_quadrature.get_weights().size();
  double             sum       = 0.;
  for (unsigned int q = 0; q < quad_size; ++q)
    {
      sum += 1.0 * JxW[q];
    }


  // Segment is inside the square, as expected the intersection is the segment
  // itself Therefore, the length of the intersection is the length of the
  // segment.
  const auto &p = cell0->vertex(0);
  const auto &q = cell0->vertex(1);
  ASSERT_DOUBLE_EQ(sum, (p - q).norm());
}
