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

#include "../source/compute_intersection_of_cells.cc"

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
    -0.5, -0.9); // move the first grid a little bit just for testing purposes
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


  const auto &       JxW       = test_quadrature.get_weights();
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



TEST(DimTester, Area_Test_Codimension0)
{
  constexpr int      dim0     = 2;
  constexpr int      dim1     = 2;
  constexpr int      spacedim = 2;
  const unsigned int degree   = 4;

  Triangulation<dim0, spacedim> tria0;
  GridGenerator::hyper_cube<dim0, spacedim>(tria0, -1., +1.);
  FE_Nothing<dim0, spacedim> dummy_fe0;
  DoFHandler<dim0, spacedim> dh0(tria0);
  dh0.distribute_dofs(dummy_fe0);
  const auto &cell0 = dh0.begin_active();
  cell0->vertex(0) =
    Point<spacedim>(-1.5, -1.4); // move the first grid a little bit
  cell0->vertex(1) = Point<spacedim>(1.3, -1.2);
  cell0->vertex(3) = Point<spacedim>(+1.3, .5);
  cell0->vertex(2) = Point<spacedim>(-.8, .2);

  // Create first polygon

  std::vector<CGAL_Point> pts0;
  pts0.emplace_back(cell0->vertex(0)[0], cell0->vertex(0)[1]); // 0
  pts0.emplace_back(cell0->vertex(1)[0], cell0->vertex(1)[1]); // 1
  pts0.emplace_back(cell0->vertex(3)[0], cell0->vertex(3)[1]); // 3
  pts0.emplace_back(cell0->vertex(2)[0], cell0->vertex(2)[1]); // 2

  const CGAL_Polygon p1(pts0.begin(), pts0.end()); // create first Polygon

  // Same thing as above
  Triangulation<dim1, spacedim> tria1;
  GridGenerator::hyper_cube<spacedim>(tria1, -1.0, 1.0);
  FE_Nothing<dim1, spacedim> dummy_fe1;
  DoFHandler<dim1, spacedim> dh1(tria1);
  dh1.distribute_dofs(dummy_fe1);
  const auto &cell1 = dh1.begin_active();

  std::vector<CGAL_Point> pts1;
  pts1.emplace_back(cell1->vertex(0)[0], cell1->vertex(0)[1]); // 0
  pts1.emplace_back(cell1->vertex(1)[0], cell1->vertex(1)[1]); // 1
  pts1.emplace_back(cell1->vertex(3)[0], cell1->vertex(3)[1]); // 3
  pts1.emplace_back(cell1->vertex(2)[0], cell1->vertex(2)[1]); // 2

  const CGAL_Polygon p2(pts1.begin(), pts1.end()); // second Polygon



  std::vector<Polygon_with_holes_2> poly_list;
  [[maybe_unused]] auto             outp_iter =
    CGAL::intersection(p1, p2, std::back_inserter(poly_list));

  const CGAL_Polygon intersect = poly_list[0].outer_boundary();

  CDT           cdt; // empty Constrained Dealanuay Triangulation object
  Vertex_handle first_vertex =
    cdt.insert(CDT::Point(CGAL::to_double(intersect.vertices_begin()->x()),
                          CGAL::to_double(intersect.vertices_begin()->y())));


  // Store vertices for each triangle
  std::vector<Vertex_handle> vec_vertices;
  for (auto it = intersect.vertices_begin(); it != intersect.vertices_end();
       ++it)
    {
      Vertex_handle va = cdt.insert(
        CDT::Point(CGAL::to_double(it->x()), CGAL::to_double(it->y())));
      vec_vertices.push_back(va);
    }


  // Now that I have coords for each vertex, I can start building the
  // triangulation, i.e. inserting the constraints
  for (unsigned int i = 0; i < vec_vertices.size() - 1; ++i)
    {
      cdt.insert_constraint(vec_vertices[i], vec_vertices[i + 1]);
    }
  cdt.insert_constraint(vec_vertices.back(),
                        first_vertex); // close the boundary of the polygon


  // Next, I compute via CGAL the area of the first element of the triangulation
  const CDT::Finite_faces_iterator it             = cdt.finite_faces_begin();
  const double                     correct_result = CGAL::to_double(
    cdt.triangle(it)
      .area()); // area of the triangle pointed to by the first iterator


  std::array<Point<dim0>, 3> vertices;
  for (unsigned int i = 0; i < 3; ++i)
    {
      vertices[i] =
        Point<dim0>{CGAL::to_double(cdt.triangle(it).vertex(i).x()),
                    CGAL::to_double(cdt.triangle(it).vertex(i).y())};
    }



  // Now construct real quadrature formula over there
  const auto quad_rule_over_triangle =
    compute_linear_transformation<dim0, dim1, 3>(QGaussSimplex<dim0>(degree),
                                                 vertices);



  // actual integration test
  const auto &       JxW     = quad_rule_over_triangle.get_weights();
  const unsigned int n_q_pts = JxW.size();
  double             sum     = 0.;
  for (unsigned int q = 0; q < n_q_pts; ++q)
    {
      sum += 1.0 * JxW[q];
    }


  EXPECT_NEAR(sum, correct_result, 1e-12);
}
