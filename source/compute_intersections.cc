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

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include "moonolith_tools.h"

using namespace dealii;

#include <set>
#include <tuple>
#include <vector>

#if defined DEAL_II_WITH_CGAL && defined DEAL_II_PREFER_CGAL_OVER_PARMOONOLITH

#  include <CGAL/Boolean_set_operations_2.h>
#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Delaunay_mesh_face_base_2.h>
#  include <CGAL/Delaunay_mesh_size_criteria_2.h>
#  include <CGAL/Delaunay_mesher_2.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#  include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Triangle_2.h>
#  include <CGAL/Triangulation_2.h>
#  include <CGAL/Triangulation_face_base_with_id_2.h>
#  include <CGAL/Triangulation_face_base_with_info_2.h>

#  include "compute_intersections.h"
#  include "compute_linear_transformation.h"

// CGAL typedefs

typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt Kernel;
typedef CGAL::Polygon_2<Kernel>            CGAL_Polygon;
typedef CGAL::Polygon_with_holes_2<Kernel> Polygon_with_holes_2;
typedef CGAL_Polygon::Point_2              CGAL_Point;
typedef CGAL_Polygon::Segment_2            CGAL_Segment;
typedef CGAL::Iso_rectangle_2<Kernel>      CGAL_Rectangle;
typedef CGAL::Triangle_2<Kernel>           CGAL_Triangle;


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

typedef CGAL::Triangulation_vertex_base_2<Kernel>                     Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, Kernel>  Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<Kernel, Fbb>      CFb;
typedef CGAL::Delaunay_mesh_face_base_2<Kernel, CFb>                  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                  Tds;
typedef CGAL::Exact_predicates_tag                                    Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds, Itag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>                      Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Face_handle   Face_handle;


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


namespace internal
{
  template <unsigned int n_vertices0 = 4, unsigned int n_vertices1 = 4>
  decltype(auto)
  compute_intersection_of_cells(
    const std::array<CGAL_Point, n_vertices0> &vertices_cell0,
    const std::array<CGAL_Point, n_vertices1> &vertices_cell1)
  {
    const CGAL_Polygon first{vertices_cell0.begin(), vertices_cell0.end()};
    const CGAL_Polygon second{vertices_cell1.begin(), vertices_cell1.end()};
    std::vector<Polygon_with_holes_2> poly_list;
    const auto                        oi =
      CGAL::intersection(first, second, std::back_inserter(poly_list));
    (void)oi;
    return poly_list;
  }

  template <>
  decltype(auto)
  compute_intersection_of_cells<4, 2>(
    const std::array<CGAL_Point, 4> &vertices_cell0,
    const std::array<CGAL_Point, 2> &vertices_cell1)
  {
    const auto first  = CGAL_Rectangle(vertices_cell0[0], vertices_cell0[3]);
    const auto second = CGAL_Segment(vertices_cell1[0], vertices_cell1[1]);
    return CGAL::intersection(first, second);
  }
} // namespace internal



namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim>
    Quadrature<spacedim>
    compute_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
      const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
      const unsigned int                                           degree,
      const Mapping<dim0, spacedim> &                              mapping0,
      const Mapping<dim1, spacedim> &                              mapping1)
    {
      Assert((dim0 != 3 || dim1 != 3 || spacedim != 3),
             ExcNotImplemented(
               "Three dimensional objects are not implemented"));

      const unsigned int n_vertices_cell0 = cell0->n_vertices();
      const unsigned int n_vertices_cell1 = cell1->n_vertices();

      const auto &deformed_vertices_cell0 = mapping0.get_vertices(
        cell0); // get deformed vertices of the current cell
      const auto &deformed_vertices_cell1 = mapping1.get_vertices(cell1);

      if (n_vertices_cell0 == 4 && n_vertices_cell1 == 4)
        { // rectangle-rectangle



          std::array<CGAL_Point, 4> vertices_cell0;

          for (unsigned int i = 0; i < 4; ++i)
            {
              vertices_cell0[i] = CGAL_Point(
                deformed_vertices_cell0[i][0],
                deformed_vertices_cell0[i][1]); // get x,y coords of the
                                                // deformed vertices
            }



          std::array<CGAL_Point, 4> vertices_cell1;


          for (unsigned int i = 0; i < n_vertices_cell1; ++i)
            {
              vertices_cell1[i] = CGAL_Point(deformed_vertices_cell1[i][0],
                                             deformed_vertices_cell1[i][1]);
            }



          std::swap(vertices_cell0[2], vertices_cell0[3]);
          std::swap(vertices_cell1[2],
                    vertices_cell1[3]); // to be consistent with dealii
          const auto inters =
            ::internal::compute_intersection_of_cells<4, 4>(vertices_cell0,
                                                            vertices_cell1);

          if (!inters.empty())
            {
              const auto &poly = inters[0].outer_boundary();

              const unsigned int size_poly = poly.size();

              if (size_poly == 4)
                {
                  std::array<Point<spacedim>, 4> vertices_array{
                    {Point<spacedim>(CGAL::to_double(poly.vertex(0).x()),
                                     CGAL::to_double(poly.vertex(0).y())),
                     Point<spacedim>(CGAL::to_double(poly.vertex(1).x()),
                                     CGAL::to_double(poly.vertex(1).y())),
                     Point<spacedim>(CGAL::to_double(poly.vertex(3).x()),
                                     CGAL::to_double(poly.vertex(3).y())),
                     Point<spacedim>(CGAL::to_double(poly.vertex(2).x()),
                                     CGAL::to_double(poly.vertex(2).y()))}};
                  return compute_linear_transformation<dim0, spacedim, 4>(
                    QGauss<dim0>(degree), vertices_array); // 4 points
                }
              else if (size_poly == 3)
                {
                  std::array<Point<spacedim>, 3> vertices_array{
                    {Point<spacedim>(CGAL::to_double(poly.vertex(0).x()),
                                     CGAL::to_double(poly.vertex(0).y())),
                     Point<spacedim>(CGAL::to_double(poly.vertex(1).x()),
                                     CGAL::to_double(poly.vertex(1).y())),
                     Point<spacedim>(CGAL::to_double(poly.vertex(2).x()),
                                     CGAL::to_double(poly.vertex(2).y()))}};
                  return compute_linear_transformation<dim0, spacedim, 3>(
                    QGaussSimplex<dim0>(degree),
                    vertices_array); // 3 points => use Quadrature for simplices
                }
              else if (size_poly > 4)
                {
                  std::pair<std::vector<Point<spacedim>>, std::vector<double>>
                    collection;

                  CDT cdt;
                  cdt.insert_constraint(poly.vertices_begin(),
                                        poly.vertices_end(),
                                        true);

                  mark_domains(cdt);
                  std::array<Point<spacedim>, 3> vertices;

                  for (Face_handle f : cdt.finite_face_handles())
                    {
                      if (f->info().in_domain() &&
                          CGAL::to_double(cdt.triangle(f).area()) > 1e-6)
                        {
                          for (unsigned int i = 0; i < 3; ++i)
                            {
                              vertices[i] = Point<spacedim>{
                                CGAL::to_double(cdt.triangle(f).vertex(i).x()),
                                CGAL::to_double(cdt.triangle(f).vertex(i).y())};
                            }



                          const auto &linear_transf =
                            compute_linear_transformation<dim0, spacedim, 3>(
                              QGaussSimplex<dim0>(degree), vertices);
                          for (const auto &pts : linear_transf.get_points())
                            collection.first.push_back(pts);
                          for (const auto &wts : linear_transf.get_weights())
                            collection.second.push_back(wts);
                        }
                    }

                  return Quadrature<spacedim>(collection.first,
                                              collection.second);
                }
            }
          else
            { // the polygon is degenerate
              return Quadrature<spacedim>();
            }
        }
      else if (n_vertices_cell0 == 4 && n_vertices_cell1 == 3)
        { // rectangle-triangle
          Assert(false,
                 ExcNotImplemented(
                   "Rectangle-Triangle intersection not yet implemented"));
        }

      else if (n_vertices_cell0 == 4 && n_vertices_cell1 == 2)
        { // segment-rectangle


          std::array<CGAL_Point, 4> vertices_cell0;

          for (unsigned int i = 0; i < 4; ++i)
            {
              vertices_cell0[i] = CGAL_Point(
                deformed_vertices_cell0[i][0],
                deformed_vertices_cell0[i][1]); // get x,y coords of the
                                                // deformed vertices
            }



          std::array<CGAL_Point, 2> vertices_cell1;

          const auto &deformed_vertices_cell1 = mapping1.get_vertices(cell1);

          for (unsigned int i = 0; i < 2; ++i)
            {
              vertices_cell1[i] = CGAL_Point(deformed_vertices_cell1[i][0],
                                             deformed_vertices_cell1[i][1]);
            }
          const auto inters =
            ::internal::compute_intersection_of_cells<4, 2>(vertices_cell0,
                                                            vertices_cell1);

          if (inters)
            {
              if (const auto *s = boost::get<CGAL_Segment>(&*inters))
                {
                  std::array<Point<spacedim>, 2> vertices_array{
                    {Point<spacedim>(CGAL::to_double(s->vertex(0).x()),
                                     CGAL::to_double(s->vertex(0).y())),
                     Point<spacedim>(CGAL::to_double(s->vertex(1).x()),
                                     CGAL::to_double(s->vertex(1).y()))}};

                  return (s->is_degenerate()) ?
                           Quadrature<spacedim>() :
                           compute_linear_transformation<dim1, spacedim, 2>(
                             QGauss<dim1>(degree),
                             vertices_array); // 2 points
                }
              else
                {
                  return Quadrature<spacedim>(); // got a simple
                                                 // Point, return an
                                                 // empty Quadrature
                }
            }
        }

      return Quadrature<spacedim>();
    }

#elif defined DEAL_II_WITH_PARMOONOLITH

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim>
    dealii::Quadrature<spacedim>
    compute_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
      const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
      const unsigned int                                           degree,
      const Mapping<dim0, spacedim> &                              mapping0,
      const Mapping<dim1, spacedim> &                              mapping1)
    {
      if constexpr ((dim0 == 1 && dim1 == 3) || (dim0 == 3 && dim1 == 1) ||
                    (dim0 == 1 && dim1 == 1))
        {
          (void)cell0;
          (void)cell1;
          (void)degree;
          (void)mapping0;
          (void)mapping1;
          AssertThrow(false, ExcNotImplemented());
          return dealii::Quadrature<spacedim>();
        }
      else
        {
          return moonolith::compute_intersection(
            cell0, cell1, degree, mapping0, mapping1);
        }
    }
#else

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim>
    Quadrature<spacedim>
    compute_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &,
      const typename Triangulation<dim1, spacedim>::cell_iterator &,
      const unsigned int,
      const Mapping<dim0, spacedim> &,
      const Mapping<dim1, spacedim> &)
    {
      Assert(false,
             ExcMessage(
               "This function needs CGAL or PARMOONOLITH to be installed, "
               "but cmake could not find any of them."));
      return Quadrature<spacedim>();
    }

#endif

    template <int dim0, int dim1, int spacedim>
    std::vector<
      std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                 typename Triangulation<dim1, spacedim>::cell_iterator,
                 Quadrature<spacedim>>>
    compute_intersection(const GridTools::Cache<dim0, spacedim> &space_cache,
                         const GridTools::Cache<dim1, spacedim> &immersed_cache,
                         const unsigned int                      degree,
                         const double                            tol)
    {
      Assert(degree >= 1, ExcMessage("degree cannot be less than 1"));

      std::vector<
        std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                   typename Triangulation<dim1, spacedim>::cell_iterator,
                   Quadrature<spacedim>>>
        cells_with_quads;


      const auto &space_tree =
        space_cache.get_locally_owned_cell_bounding_boxes_rtree();

      // The immersed tree *must* contain all cells, also the non-locally owned
      // ones.
      const auto &immersed_tree =
        immersed_cache.get_cell_bounding_boxes_rtree();

      // references to triangulations' info (cp cstrs marked as delete)
      const auto &mapping0 = space_cache.get_mapping();
      const auto &mapping1 = immersed_cache.get_mapping();
      namespace bgi        = boost::geometry::index;
      // Whenever the BB space_cell intersects the BB of an embedded cell,
      // store the space_cell in the set of intersected_cells
      for (const auto &[immersed_box, immersed_cell] : immersed_tree)
        {
          for (const auto &[space_box, space_cell] :
               space_tree |
                 bgi::adaptors::queried(bgi::intersects(immersed_box)))
            {
              const auto test_intersection =
                compute_intersection<dim0, dim1, spacedim>(
                  space_cell, immersed_cell, degree, mapping0, mapping1);

              // if (test_intersection.get_points().size() !=
              const auto & weights = test_intersection.get_weights();
              const double area =
                std::accumulate(weights.begin(), weights.end(), 0.0);
              if (area > tol) // non-trivial intersection
                {
                  cells_with_quads.push_back(std::make_tuple(
                    space_cell, immersed_cell, test_intersection));
                }
            }
        }

      return cells_with_quads;
    }


    template Quadrature<1>
    compute_intersection(const Triangulation<1, 1>::cell_iterator &,
                         const Triangulation<1, 1>::cell_iterator &,
                         const unsigned int,
                         const Mapping<1, 1> &,
                         const Mapping<1, 1> &);



    template Quadrature<2>
    compute_intersection(const Triangulation<2, 2>::cell_iterator &,
                         const Triangulation<1, 2>::cell_iterator &,
                         const unsigned int,
                         const Mapping<2, 2> &,
                         const Mapping<1, 2> &);

    template Quadrature<2>
    compute_intersection(const Triangulation<2, 2>::cell_iterator &,
                         const Triangulation<2, 2>::cell_iterator &,
                         const unsigned int,
                         const Mapping<2, 2> &,
                         const Mapping<2, 2> &);


    template Quadrature<3>
    compute_intersection(const Triangulation<3, 3>::cell_iterator &,
                         const Triangulation<2, 3>::cell_iterator &,
                         const unsigned int,
                         const Mapping<3, 3> &,
                         const Mapping<2, 3> &);

    template Quadrature<3>
    compute_intersection(const Triangulation<3, 3>::cell_iterator &,
                         const Triangulation<3, 3>::cell_iterator &,
                         const unsigned int,
                         const Mapping<3, 3> &,
                         const Mapping<3, 3> &);

    template std::vector<
      std::tuple<typename dealii::Triangulation<1, 1>::cell_iterator,
                 typename dealii::Triangulation<1, 1>::cell_iterator,
                 Quadrature<1>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<1, 1> &space_cache,
      const GridTools::Cache<1, 1> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<
      std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
                 typename dealii::Triangulation<1, 3>::cell_iterator,
                 Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<1, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);


    template std::vector<
      std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                 typename dealii::Triangulation<1, 2>::cell_iterator,
                 Quadrature<2>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<2, 2> &space_cache,
      const GridTools::Cache<1, 2> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<2, 2>::cell_iterator,
                                    typename Triangulation<2, 2>::cell_iterator,
                                    Quadrature<2>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<2, 2> &space_cache,
      const GridTools::Cache<2, 2> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                    typename Triangulation<2, 3>::cell_iterator,
                                    Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<2, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);


    template std::vector<std::tuple<typename Triangulation<2, 3>::cell_iterator,
                                    typename Triangulation<3, 3>::cell_iterator,
                                    Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<2, 3> &space_cache,
      const GridTools::Cache<3, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                    typename Triangulation<3, 3>::cell_iterator,
                                    Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<3, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);
  }
}
