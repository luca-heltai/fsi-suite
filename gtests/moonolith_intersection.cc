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

#  include "dim_spacedim_tester.h"
#  include "moonolith_convex_decomposition.hpp"
#  include "moonolith_intersect_polyhedra.hpp"
#  include "moonolith_mesh_io.hpp"
#  include "moonolith_par_l2_transfer.hpp"
#  include "par_moonolith.hpp"

using namespace dealii;
using namespace moonolith;


template <int spacedim, int dim, class T1, class T2>
Quadrature<double, spacedim>
compute_intersection(const Quadrature<double, dim> &ref_quad,
                     const T1 &                     t1,
                     const T2 &                     t2)
{
  BuildQuadrature<T1, T2>      intersect;
  Quadrature<double, spacedim> out;
  intersect.apply(ref_quad, t1, t2, out);
  return out;
}

TEST(MoonoLith, CheckSquare)
{
  ASSERT_TRUE(true);

  // Intersection of two squares and quadrature

  Polygon<Real, 2> poly1;
  Polygon<Real, 2> poly2;
  // Polygon<Real,2> Resultint;

  poly1.points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  poly2.points = {{0.5, 0.5}, {1.5, 0.5}, {1.5, 1.5}, {0.5, 1.5}};

  // int dim = 2;
  // int spacedim = 2;

  // BuildQuadrature<poly1,>
  Quadrature<Real, 2> ref_quad;

  ASSERT_TRUE(moonolith::Gauss::get(2, ref_quad));
  auto refintergral = moonolith::measure(
    ref_quad); // std::accumulate(ref_quad.weights.begin(),ref_quad.weights.end(),0);

  ASSERT_NEAR(refintergral, 1, 1e-10);

  auto quad = compute_intersection<2>(ref_quad, poly1, poly2);

  auto intergral = moonolith::measure(
    quad); //::accumulate(quad.weights.begin(),quad.weights.end(),0);

  ASSERT_NEAR(intergral, 0.25, 1e-10);
}



TEST(MoonoLith, CheckTetra)
{
  Polyhedron<Real> poly1, poly2, inter;
  bool             inter_check;

  poly1.el_ptr   = {0, 3, 6, 9, 12};                     //
  poly1.el_index = {0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3}; // faces
  poly1.points   = {{0.0, 0.0, 0.0},
                  {1.0, 0.0, 0.0},
                  {0.0, 1.0, 0.0},
                  {0.0, 0.0, 1.0}};

  poly2        = poly1;
  poly2.points = {{0.0, 0.0, 0.0},
                  {0.5, 0.0, 0.0},
                  {0.0, 0.5, 0.0},
                  {0.0, 0.0, 0.5}}; // poly2 is immersed in poly1

  poly1.fix_ordering();
  poly2.fix_ordering();

  auto vol1      = measure(poly1);
  auto tetravol1 = tetrahedron_volume(poly1.points[0],
                                      poly1.points[1],
                                      poly1.points[2],
                                      poly1.points[3]);
  ASSERT_NEAR(vol1, tetravol1, 1e-10);

  auto vol2      = measure(poly2);
  auto tetravol2 = tetrahedron_volume(poly2.points[0],
                                      poly2.points[1],
                                      poly2.points[2],
                                      poly2.points[3]);
  ASSERT_NEAR(vol2, tetravol2, 1e-10);

  inter_check = moonolith::intersect_convex_polyhedra(poly1, poly2, inter);
  ASSERT_TRUE(inter_check);

  MatlabScripter plot;
  plot.plot(poly1);
  plot.plot(poly2);
  plot.plot(inter);
  plot.save("Test1.m");

  Quadrature<Real, 3> ref_quad;

  ASSERT_TRUE(moonolith::Gauss::get(2, ref_quad));

  auto refintergral = moonolith::measure(ref_quad);

  auto v = ref_quad.weights;
  for (int d = 0; d < 4; ++d)
    {
      ASSERT_NEAR(v[d], 0.025, 1e-10);
    }
  for (int d = 4; d < 8; ++d)
    {
      ASSERT_NEAR(v[d], 0.225, 1e-10);
    }

  ASSERT_NEAR(refintergral, 1, 1e-10);

  auto quad = compute_intersection<3>(ref_quad, poly1, poly2);

  auto integral = moonolith::measure(quad);

  ASSERT_NEAR(integral, vol2, 1e-10);
}



TEST(MoonoLith, CheckNonTrivialTetraInt)
{
  Polyhedron<Real> poly1, poly2, inter;

  poly1.el_ptr   = {0, 3, 6, 9, 12};
  poly1.el_index = {0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 3, 2};
  poly1.points   = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  poly2        = poly1;
  poly2.points = {{-0.75, 0.25, 0.5},
                  {0.25, 0.25, 0.5},
                  {0.25, -0.75, 0.5},
                  {0.25, 0.25, -0.5}};

  poly1.fix_ordering();
  poly2.fix_ordering();

  auto vol1 = measure(poly1);
  ASSERT_NEAR(vol1, 1. / 6., 1e-10);
  auto vol2 = measure(poly2);
  ASSERT_NEAR(vol2, 1. / 6., 1e-10);

  auto inter_check = moonolith::intersect_convex_polyhedra(poly1, poly2, inter);
  ASSERT_TRUE(inter_check);
  const auto vol_inter = moonolith::measure(inter);
  ASSERT_NEAR(vol_inter, 1. / 32., 1e-10);

  Quadrature<Real, 3> ref_quad;

  ASSERT_TRUE(moonolith::Gauss::get(1, ref_quad));

  // MatlabScripter plot;
  // plot.plot(poly1);
  // plot.plot(poly2);
  // plot.plot(inter);
  // plot.save("Test.m");

  auto refintegral = moonolith::measure(ref_quad);
  ASSERT_NEAR(refintegral, 1, 1e-10);

  auto quad = compute_intersection<3>(ref_quad, poly1, poly2);

  auto integral = moonolith::measure(quad);
  ASSERT_NEAR(integral, vol_inter, 1e-10);
}

TEST(MoonoLith, CheckExahedra)
{
  Polyhedron<Real> poly1, poly2, poly3, inter;

  // Unit cube
  poly1.el_ptr   = {0, 4, 8, 12, 16, 20, 24};
  poly1.el_index = {0, 1, 2, 3, 1, 2, 6, 5, 2, 6, 7, 3,
                    0, 3, 7, 4, 0, 1, 5, 4, 4, 5, 6, 7};
  poly1.points   = {{0, 0, 0},
                  {1, 0, 0},
                  {1, 1, 0},
                  {0, 1, 0},
                  {0, 0, 1},
                  {1, 0, 1},
                  {1, 1, 1},
                  {0, 1, 1}};
  poly1.fix_ordering();
  auto vol1 = moonolith::measure(poly1);
  ASSERT_NEAR(vol1, 1, 1e-10);

  // Shift of Poly 1 by x = 0.5
  poly2        = poly1;
  poly2.points = {{0.5, 0, 0},
                  {1.5, 0, 0},
                  {1.5, 1, 0},
                  {0.5, 1, 0},
                  {0.5, 0, 1},
                  {1.5, 0, 1},
                  {1.5, 1, 1},
                  {0.5, 1, 1}};
  poly2.fix_ordering();
  auto vol2 = moonolith::measure(poly2);
  ASSERT_NEAR(vol2, 1, 1e-10);

  // poly3 is a deformation of poly2 - Exahedron
  poly3        = poly1;
  poly3.points = {{0, 0, 0},
                  {1.5, 0, 0},
                  {1.5, 1, 0},
                  {0.5, 1, 0},
                  {0.5, 0, 1},
                  {1.5, 0, 1},
                  {1.5, 1, 1},
                  {0.5, 1, 1}};
  poly3.fix_ordering();
  auto vol3 = moonolith::measure(poly3);
  ASSERT_NEAR(vol3, 1.0833333333333333, 1e-10);

  // Intersection between the Unit Square and the exahedron poly2
  auto inter_check = moonolith::intersect_convex_polyhedra(poly1, poly2, inter);
  ASSERT_TRUE(inter_check);

  // Check intersection measure
  auto volint = moonolith::measure(inter);
  ASSERT_NEAR(volint, 0.5, 1e-10);

  MatlabScripter plot;
  plot.plot(poly1);
  plot.plot(poly3);
  plot.plot(inter);
  plot.save("ExaTest.m");

  Quadrature<Real, 3> ref_quad;
  moonolith::Gauss::get(1, ref_quad);
  auto quad     = compute_intersection<3>(ref_quad, poly1, poly2);
  auto integral = moonolith::measure(quad);
  ASSERT_NEAR(integral, 0.5, 1e-10);
}

// The following test fails. ParMoonolith does not support non planar faces.
TEST(MoonoLith, DISABLED_CheckExahedraNonPlanar)
{
  Polyhedron<Real> poly1, poly2, inter;

  make_cube({0, 0, 0}, {1, 1, 1}, poly1);
  const auto vol1 = moonolith::measure(poly1);

  make_cube({0.5, 0, 0}, {1.5, 1, 1}, poly2);
  ASSERT_NEAR(vol1, 1, 1e-10);

  const auto vol2 = moonolith::measure(poly2);
  ASSERT_NEAR(vol2, 1, 1e-10);

  // Deform poly2
  poly2.points[0] = {0, 0, 0};
  auto vol3       = moonolith::measure(poly2);
  ASSERT_NEAR(vol3, 1.0833333333333333, 1e-10);

  // Intersection between the Unit Square and the exahedron poly2
  auto inter_check = moonolith::intersect_convex_polyhedra(poly1, poly2, inter);
  ASSERT_TRUE(inter_check);

  // Check intersection measure
  auto volint = moonolith::measure(inter);
  ASSERT_NEAR(volint, 0.583333333333333333, 1e-10);

  MatlabScripter plot;
  plot.plot(poly1);
  plot.plot(poly2);
  plot.plot(inter);
  plot.save("ExaTest.m");

  Quadrature<Real, 3> ref_quad;
  moonolith::Gauss::get(1, ref_quad);
  auto quad     = compute_intersection<3>(ref_quad, poly1, poly2);
  auto integral = moonolith::measure(quad);
  ASSERT_NEAR(integral, 1.0833333333333333 - 0.5, 1e-10);
}

#endif