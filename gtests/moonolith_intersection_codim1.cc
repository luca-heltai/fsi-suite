#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <moonolith_build_quadrature.hpp>
#  include <moonolith_polygon.hpp>

#  include <array>
#  include <cassert>
#  include <cmath>
#  include <fstream>
#  include <iostream>
#  include <sstream>

#  include "deal_moon_conversion.hpp"
#  include "dim_spacedim_tester.h"
#  include "moonolith_convex_decomposition.hpp"
#  include "moonolith_intersect_polyhedra.hpp"
#  include "moonolith_mesh_io.hpp"
#  include "moonolith_par_l2_transfer.hpp"
#  include "par_moonolith.hpp"

using namespace moonolith;


template <int spacedim, int dim, class T1, class T2>
moonolith::Quadrature<double, spacedim>
compute_intersection(const moonolith::Quadrature<double, dim> &ref_quad,
                     const T1 &                                t1,
                     const T2 &                                t2)
{
  BuildQuadrature<T1, T2>                 intersect;
  moonolith::Quadrature<double, spacedim> out;
  intersect.apply(ref_quad, t1, t2, out);
  return out;
}

TEST(MoonoLith, PolyLine)
{
  ASSERT_TRUE(true);

  // Intersection of a square and a line

  Polygon<Real, 2> poly;
  Line<double, 2>  line;

  poly.points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  moonolith::Vector<double, 2> pts0{-.5, .5};
  moonolith::Vector<double, 2> pts1{1.5, .5};
  line.p0 = pts0;
  line.p1 = pts1;

  moonolith::Quadrature<Real, 1> ref_quad;

  ASSERT_TRUE(moonolith::Gauss::get(2, ref_quad));
  auto refintegral = moonolith::measure(ref_quad);

  ASSERT_NEAR(refintegral, 1, 1e-10);

  auto quad = compute_intersection<2>(ref_quad, poly, line);

  auto integral = std::accumulate(quad.weights.begin(), quad.weights.end(), 0.);
  ASSERT_NEAR(integral, 1.0, 1e-10);
}



TEST(MoonoLith, PolyLineInside)
{
  ASSERT_TRUE(true);

  // Intersection of a square and a line

  Polygon<Real, 2> poly;
  Line<double, 2>  line;

  poly.points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  moonolith::Vector<double, 2> pts0{.2, .3};
  moonolith::Vector<double, 2> pts1{.8, .9};
  line.p0 = pts0;
  line.p1 = pts1;
  const double expected =
    std::sqrt(std::pow(pts0[0] - pts1[0], 2) + std::pow(pts0[1] - pts1[1], 2));

  moonolith::Quadrature<Real, 1> ref_quad;

  ASSERT_TRUE(moonolith::Gauss::get(2, ref_quad));



  auto quad = compute_intersection<2>(ref_quad, poly, line);

  auto integral = std::accumulate(quad.weights.begin(), quad.weights.end(), 0.);
  ASSERT_NEAR(integral, expected, 1e-10);
}



#endif