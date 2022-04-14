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

#ifdef DEAL_II_WITH_CGAL

#  include <deal.II/dofs/dof_handler.h>

#  include <deal.II/fe/fe_nothing.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/tria.h>


// CGAL headers and typedefs
#  include <CGAL/Boolean_set_operations_2.h>
#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Delaunay_mesh_face_base_2.h>
#  include <CGAL/Delaunay_mesh_size_criteria_2.h>
#  include <CGAL/Delaunay_mesher_2.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Triangulation_2.h>
#  include <CGAL/Triangulation_face_base_with_id_2.h>
#  include <CGAL/Triangulation_face_base_with_info_2.h>

#  include "compute_intersections.h"
#  include "compute_linear_transformation.h"
#  include "dim_spacedim_tester.h"

struct FaceInfo2
{
  FaceInfo2()
  {}
  int nesting_level;
  bool
  in_domain()
  {
    return nesting_level % 2 == 1;
  }
};


typedef CGAL::Exact_predicates_exact_constructions_kernel        K;
typedef CGAL::Triangulation_vertex_base_2<K>                     Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K>  Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<K, Fbb>      CFb;
typedef CGAL::Delaunay_mesh_face_base_2<K, CFb>                  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>             TDS;
typedef CGAL::Exact_predicates_tag                               Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>                 Criteria;
typedef CDT::Point                                               CGAL_Point;
typedef CGAL::Polygon_2<K>                                       CGAL_Polygon;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes_2;
typedef CDT::Face_handle              Face_handle;



void
mark_domains(CDT &                 ct,
             Face_handle           start,
             int                   index,
             std::list<CDT::Edge> &border)
{
  if (start->info().nesting_level != -1)
    {
      return;
    }
  std::list<Face_handle> queue;
  queue.push_back(start);
  while (!queue.empty())
    {
      Face_handle fh = queue.front();
      queue.pop_front();
      if (fh->info().nesting_level == -1)
        {
          fh->info().nesting_level = index;
          for (int i = 0; i < 3; i++)
            {
              CDT::Edge   e(fh, i);
              Face_handle n = fh->neighbor(i);
              if (n->info().nesting_level == -1)
                {
                  if (ct.is_constrained(e))
                    border.push_back(e);
                  else
                    queue.push_back(n);
                }
            }
        }
    }
}

void
mark_domains(CDT &cdt)
{
  for (CDT::Face_handle f : cdt.all_face_handles())
    {
      f->info().nesting_level = -1;
    }
  std::list<CDT::Edge> border;
  mark_domains(cdt, cdt.infinite_face(), 0, border);
  while (!border.empty())
    {
      CDT::Edge e = border.front();
      border.pop_front();
      Face_handle n = e.first->neighbor(e.second);
      if (n->info().nesting_level == -1)
        {
          mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
        }
    }
}



using namespace dealii;

TEST(CGAL, BooleanOperationsPolygons)
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


  std::vector<Polygon_with_holes_2> diff_poly;
  [[maybe_unused]] const auto       outer_poly_it =
    CGAL::difference(p1, intersect, std::back_inserter(diff_poly));
  const auto outer_poly = diff_poly[0].outer_boundary();

  CDT cdt; // empty Constrained Dealanuay Triangulation object
  cdt.insert_constraint(outer_poly.vertices_begin(),
                        outer_poly.vertices_end(),
                        true);


  // Next, I compute via CGAL the area of the polygon
  const double correct_result = CGAL::to_double(CGAL::polygon_area_2(
    outer_poly.vertices_begin(),
    outer_poly.vertices_end(),
    K())); // area of the triangle pointed to by the first iterator

  mark_domains(cdt);


  std::array<Point<dim0>, 3> vertices;
  double                     sum = 0.;
  // Loop over finite faces (or "cells" in dealii), get the area of that element
  // and sum it.
  for (Face_handle f : cdt.finite_face_handles())
    {
      if (f->info().in_domain())
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              vertices[i] =
                Point<dim0>{CGAL::to_double(cdt.triangle(f).vertex(i).x()),
                            CGAL::to_double(cdt.triangle(f).vertex(i).y())};
            }



          // Now construct real quadrature formula over
          //   there
          const auto quad_rule_over_triangle =
            compute_linear_transformation<dim0, dim1, 3>(
              QGaussSimplex<dim0>(degree), vertices);



          // actual integration test
          const auto &       JxW     = quad_rule_over_triangle.get_weights();
          const unsigned int n_q_pts = JxW.size();
          for (unsigned int q = 0; q < n_q_pts; ++q)
            {
              sum += 1.0 * JxW[q];
            }
        }
    }

  EXPECT_NEAR(sum, correct_result, 1e-12);
}
#endif