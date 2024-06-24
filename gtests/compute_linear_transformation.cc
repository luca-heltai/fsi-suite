// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Luca Heltai
//
// This file is part of the FSI-suite platform, based on the deal.II library.
//
// The FSI-suite platform is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3.0 of theLicense,
// or (at your option) any later version. The full text of the license can be
// found in the file LICENSE at the top level of the FSI-suite platform
// distribution.
//
// ---------------------------------------------------------------------

#include "compute_linear_transformation.h"

#include <fstream>
#include <sstream>

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

template <int degree>
struct Test_polynomial
{
  double
  operator()(const Point<2> &p)
  {
    return std::pow(p[0], degree) + std::pow(p[1], degree);
  }
};

TEST(DimTester, IntegralOverLine)
{
  constexpr int dim      = 1;
  constexpr int spacedim = 2;
  constexpr int N        = 2;



  std::array<Point<spacedim>, 2> vertices{
    {Point<spacedim>(0.0, 2.0), Point<spacedim>(1.0, 0.0)}};
  const double correct_result =
    2.3261866736701; // Integral of x+y-sin(x) over the line from (0,2) to (1,0)
  QGauss<dim> quad(4);
  const auto  real_quadrature =
    compute_linear_transformation<dim, spacedim, N>(quad, vertices);


  Test_function      fun{};
  double             sum            = 0.;
  const auto        &quad_points    = real_quadrature.get_points();
  const auto        &JxW            = real_quadrature.get_weights();
  const unsigned int size_quad_form = quad_points.size();
  for (unsigned int q = 0; q < size_quad_form; ++q)
    {
      sum += fun(quad_points[q]) * JxW[q];
    }

  EXPECT_NEAR(sum, correct_result, 1e-12);
}


TEST(DimTester, IntegrationTestCodimension0)
{
  constexpr unsigned int dim0   = 2;
  constexpr unsigned int dim1   = 2;
  constexpr unsigned int degree = 4;


  // Here we integarte over the triangle given by the following vertices.
  // In practice, we have x \in [0,4] and y \in [8-2x]
  std::array<Point<dim0>, 3> vertices{
    {Point<dim0>{0., 0.}, Point<dim0>{4., 0.}, Point<dim0>{0., 8.}}};


  // The correct value is computed using
  // https://www.wolframalpha.com/widgets/view.jsp?id=f5f3cbf14f4f5d6d2085bf2d0fb76e8a
  const double correct_result = 69632. / 15.;

  const auto quad_rule_over_triangle =
    compute_linear_transformation<dim0, dim1, 3>(QGaussSimplex<dim0>(degree),
                                                 vertices);



  // Actual integration test. The template parameter degree determines the
  // exponent of the polynomial function to be integrated, and the exactness of
  // the quadrature formula
  Test_polynomial<degree> func;
  const auto             &xq      = quad_rule_over_triangle.get_points();
  const auto             &JxW     = quad_rule_over_triangle.get_weights();
  const unsigned int      n_q_pts = JxW.size();
  double                  sum     = 0.;
  for (unsigned int q = 0; q < n_q_pts; ++q)
    {
      sum += func(xq[q]) * JxW[q];
    }


  EXPECT_NEAR(sum, correct_result, 1e-12);
}