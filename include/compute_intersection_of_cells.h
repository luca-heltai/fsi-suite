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


#ifndef compute_intersction_of_cells_h
#define compute_intersction_of_cells_h


// CGAL header files and typedefs
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/Triangulation_2.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt Kernel;
typedef CGAL::Polygon_2<Kernel>            CGAL_Polygon;
typedef CGAL::Polygon_with_holes_2<Kernel> Polygon_with_holes_2;
typedef CGAL_Polygon::Point_2              CGAL_Point;
typedef CGAL_Polygon::Segment_2            CGAL_Segment;
typedef CGAL::Iso_rectangle_2<Kernel>      CGAL_Rectangle;
typedef CGAL::Triangle_2<Kernel>           CGAL_Triangle;

// typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt K;
typedef CGAL::Triangulation_vertex_base_2<K>                        Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K>                          Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds>          CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>                    Criteria;
typedef CDT::Vertex_handle Vertex_handle;



#include "compute_linear_transformation.h"



template <unsigned int vertices0 = 4, unsigned int vertices1 = 4>
decltype(auto)
compute_intersection_of_cells(const std::vector<CGAL_Point> &vertices_cell0,
                              const std::vector<CGAL_Point> &vertices_cell1)
{
  std::cout << "Function called" << '\n';
  const auto first  = CGAL_Rectangle(vertices_cell0[0], vertices_cell0[3]);
  const auto second = CGAL_Rectangle(vertices_cell1[0], vertices_cell1[3]);
  return CGAL::intersection(first, second);
}

template <>
decltype(auto)
compute_intersection_of_cells<2, 4>(
  const std::vector<CGAL_Point> &vertices_cell0,
  const std::vector<CGAL_Point> &vertices_cell1)
{
  const auto first  = CGAL_Segment(vertices_cell0[0], vertices_cell0[1]);
  const auto second = CGAL_Rectangle(vertices_cell1[0], vertices_cell1[3]);
  return CGAL::intersection(first, second);
}

/**
 * @brief Intersect `cell0` and `cell1` and construct a `Quadrature<spacedim>` of degree `degree``
 *        over the intersection. Mappings for both cells are in `mapping0` and
 * `mapping1`, respectively.
 *
 * @tparam dim0
 * @tparam dim1
 * @tparam spacedim
 * @param cell0
 * @param cell1
 * @param degree
 * @param mapping0
 * @param mapping1
 * @return Quadrature<spacedim>
 */
template <int dim0, int dim1, int spacedim>
Quadrature<spacedim>
compute_intersection(
  const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
  const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
  const unsigned int                                           degree,
  const Mapping<dim0, spacedim>                               &mapping0 =
    (ReferenceCells::get_hypercube<dim0>()
       .template get_default_linear_mapping<dim0, spacedim>()),
  const Mapping<dim1, spacedim> &mapping1 =
    (ReferenceCells::get_hypercube<dim1>()
       .template get_default_linear_mapping<dim1, spacedim>()))
{
  Assert((dim0 != 3 | dim1 != 3 | spacedim != 3),
         ExcNotImplemented("Three dimensional objects are not implemented"));

  const unsigned int      n_vertices_cell0 = cell0->n_vertices();
  std::vector<CGAL_Point> vertices_cell0(n_vertices_cell0);

  const auto &deformed_vertices_cell0 =
    mapping0.get_vertices(cell0); // get deformed vertices of the current cell

  // collect vertices of cell0 as CGAL_Point(s)
  for (unsigned int i = 0; i < n_vertices_cell0; ++i)
    {
      vertices_cell0[i] =
        CGAL_Point(deformed_vertices_cell0[i][0],
                   deformed_vertices_cell0[i][1]); // get x,y coords of the
                                                   // deformed vertices
    }

  const unsigned int      n_vertices_cell1 = cell1->n_vertices();
  std::vector<CGAL_Point> vertices_cell1(n_vertices_cell1);
  const auto &deformed_vertices_cell1 = mapping1.get_vertices(cell1);
  for (unsigned int i = 0; i < n_vertices_cell1; ++i)
    {
      vertices_cell1[i] = CGAL_Point(deformed_vertices_cell1[i][0],
                                     deformed_vertices_cell1[i][1]);
    }


  if (n_vertices_cell0 == 4 && n_vertices_cell1 == 4)
    { // rectangle-rectangle
      auto inters =
        compute_intersection_of_cells<4, 4>(vertices_cell0, vertices_cell1);

      if (inters)
        {
          const auto *r = boost::get<CGAL_Rectangle>(&*inters);
          std::cout << *r << '\n'; // TODO
          assert(!r->is_degenerate());
          std::array<Point<spacedim>, 4> vertices_array{
            Point<spacedim>(CGAL::to_double(r->vertex(0).x()),
                            CGAL::to_double(r->vertex(0).y())),
            Point<spacedim>(CGAL::to_double(r->vertex(1).x()),
                            CGAL::to_double(r->vertex(1).y())),
            Point<spacedim>(CGAL::to_double(r->vertex(2).x()),
                            CGAL::to_double(r->vertex(2).y())),
            Point<spacedim>(CGAL::to_double(r->vertex(3).x()),
                            CGAL::to_double(r->vertex(3).y()))};

          return compute_linear_transformation<dim0, dim1, 4>(
            QGauss<dim0>(degree), vertices_array); // 4 points
        }
    }
  else if (n_vertices_cell0 == 4 && n_vertices_cell1 == 3)
    { // rectangle-triangle
      ExcNotImplemented("Rectangle-Triangle intersection not yet implemented");
    }

  else if (n_vertices_cell0 == 2 && n_vertices_cell1 == 4)
    { // segment-rectangle
      auto inters =
        compute_intersection_of_cells<2, 4>(vertices_cell0, vertices_cell1);

      if (inters)
        {
          if (const auto *s = boost::get<CGAL_Segment>(&*inters))
            {
              std::array<Point<spacedim>, 2> vertices_array{
                Point<spacedim>(CGAL::to_double(s->vertex(0).x()),
                                CGAL::to_double(s->vertex(0).y())),
                Point<spacedim>(CGAL::to_double(s->vertex(1).x()),
                                CGAL::to_double(s->vertex(1).y()))};

              return (s->is_degenerate()) ?
                       Quadrature<spacedim>() :
                       compute_linear_transformation<dim0, spacedim, 2>(
                         QGauss<dim0>(degree),
                         vertices_array); // 2 points
            }
          else
            {
              return Quadrature<spacedim>(); // got a simple Point, return an
                                             // empty Quadrature
            }
        }
    }

  return Quadrature<spacedim>();
}


#endif