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


#ifdef DEAL_II_WITH_CGAL

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

#  include <set>
#  include <tuple>
#  include <vector>

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

// typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt K;
typedef CGAL::Triangulation_vertex_base_2<K>                        Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K>                          Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds>          CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>                    Criteria;
typedef CDT::Vertex_handle Vertex_handle;

namespace internal
{
  template <unsigned int vertices0 = 4, unsigned int vertices1 = 4>
  decltype(auto)
  compute_intersection_of_cells(const std::vector<CGAL_Point> &vertices_cell0,
                                const std::vector<CGAL_Point> &vertices_cell1)
  {
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
} // namespace internal


using namespace dealii;

namespace dealii::NonMatching
{
  template <int dim0, int dim1, int spacedim>
  dealii::Quadrature<spacedim>
  compute_intersection(
    const typename dealii::Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename dealii::Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                                   degree,
    const dealii::Mapping<dim0, spacedim> &mapping0,
    const dealii::Mapping<dim1, spacedim> &mapping1)
  {
    Assert((dim0 != 3 || dim1 != 3 || spacedim != 3),
           dealii::ExcNotImplemented(
             "Three dimensional objects are not implemented"));

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
        const auto inters =
          ::internal::compute_intersection_of_cells<4, 4>(vertices_cell0,
                                                          vertices_cell1);

        if (inters)
          {
            if (const auto *r = boost::get<CGAL_Rectangle>(&*inters))
              {
                // std::cout << *r << '\n'; // TODO
                // assert(!r->is_degenerate());
                std::array<dealii::Point<spacedim>, 4> vertices_array{
                  {dealii::Point<spacedim>(CGAL::to_double(r->vertex(0).x()),
                                           CGAL::to_double(r->vertex(0).y())),
                   dealii::Point<spacedim>(CGAL::to_double(r->vertex(1).x()),
                                           CGAL::to_double(r->vertex(1).y())),
                   dealii::Point<spacedim>(CGAL::to_double(r->vertex(3).x()),
                                           CGAL::to_double(r->vertex(3).y())),
                   dealii::Point<spacedim>(CGAL::to_double(r->vertex(2).x()),
                                           CGAL::to_double(r->vertex(2).y()))}};


                return (r->is_degenerate()) ?
                         dealii::Quadrature<spacedim>() :
                         compute_linear_transformation<dim0, spacedim, 4>(
                           dealii::QGauss<dim0>(degree),
                           vertices_array); // 4 points
              }
            else
              {
                return dealii::Quadrature<spacedim>();
              }
          }
      }
    else if (n_vertices_cell0 == 4 && n_vertices_cell1 == 3)
      { // rectangle-triangle
        dealii::ExcNotImplemented(
          "Rectangle-Triangle intersection not yet implemented");
      }

    else if (n_vertices_cell0 == 2 && n_vertices_cell1 == 4)
      { // segment-rectangle
        const auto inters =
          ::internal::compute_intersection_of_cells<2, 4>(vertices_cell0,
                                                          vertices_cell1);

        if (inters)
          {
            if (const auto *s = boost::get<CGAL_Segment>(&*inters))
              {
                std::array<dealii::Point<spacedim>, 2> vertices_array{
                  {dealii::Point<spacedim>(CGAL::to_double(s->vertex(0).x()),
                                           CGAL::to_double(s->vertex(0).y())),
                   dealii::Point<spacedim>(CGAL::to_double(s->vertex(1).x()),
                                           CGAL::to_double(s->vertex(1).y()))}};

                return (s->is_degenerate()) ?
                         dealii::Quadrature<spacedim>() :
                         compute_linear_transformation<dim0, spacedim, 2>(
                           dealii::QGauss<dim0>(degree),
                           vertices_array); // 2 points
              }
            else
              {
                return dealii::Quadrature<spacedim>(); // got a simple Point,
                                                       // return an empty
                                                       // Quadrature
              }
          }
      }

    return dealii::Quadrature<spacedim>();
  }

  template <int dim0, int dim1, int spacedim>
  std::vector<std::tuple<
    typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
    typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
    dealii::Quadrature<spacedim>>>
  compute_intersection(const GridTools::Cache<dim0, spacedim> &space_cache,
                       const GridTools::Cache<dim1, spacedim> &immersed_cache,
                       const unsigned int                      degree)
  {
    Assert(DEAL_II_WITH_CGAL,
           ExcMessage("This function needs CGAL to be installed, "
                      "but cmake could not find it."));

    Assert(degree >= 1, ExcMessage("degree cannot be 0"));

    std::set<typename Triangulation<dim1, spacedim>::active_cell_iterator>
                intersected_cells; // avoid duplicates
    const auto &space_tree =
      space_cache.get_locally_owned_cell_bounding_boxes_rtree();
    const auto &immersed_tree =
      immersed_cache.get_locally_owned_cell_bounding_boxes_rtree();


    namespace bgi = boost::geometry::index;
    // Whenever the BB space_cell intersects the BB of an embedded cell, store
    // the space_cell in the set of intersected_cells
    for (const auto &[immersed_box, immersed_cell] : immersed_tree)
      {
        for (const auto &[space_box, space_cell] :
             space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box)))
          {
            intersected_cells.insert(space_cell);
          }
      } // found intersected cells


    // references to triangulations' info (cp cstrs marked as delete)
    const auto &mapping0 = space_cache.get_mapping();
    const auto &mapping1 = immersed_cache.get_mapping();

    std::vector<
      std::tuple<typename Triangulation<dim0, spacedim>::active_cell_iterator,
                 typename Triangulation<dim1, spacedim>::active_cell_iterator,
                 Quadrature<spacedim>>>
      cells_with_quads;

    for (const auto &space_cell : intersected_cells)
      { // loop over interseced space_cells
        for (const auto &[immersed_box, immersed_cell] : immersed_tree)
          {
            typename Triangulation<dim0, spacedim>::active_cell_iterator
              space_cell_t(space_cell);

            typename Triangulation<dim1, spacedim>::active_cell_iterator
              immersed_cell_t(immersed_cell);

            const auto test_intersection =
              compute_intersection<dim0, dim1, spacedim>(
                space_cell_t, immersed_cell_t, degree, mapping0, mapping1);

            if (test_intersection.get_points().size() != 0)
              {
                cells_with_quads.push_back(std::make_tuple(space_cell_t,
                                                           immersed_cell_t,
                                                           test_intersection));
              }
          }
      }

    return cells_with_quads;
  }

} // namespace dealii::NonMatching



template dealii::Quadrature<1>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<1, 1>::cell_iterator &,
  const dealii::Triangulation<1, 1>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<1, 1> &,
  const dealii::Mapping<1, 1> &);

template dealii::Quadrature<2>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<1, 2>::cell_iterator &,
  const dealii::Triangulation<1, 2>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<1, 2> &,
  const dealii::Mapping<1, 2> &);

template dealii::Quadrature<2>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<1, 2>::cell_iterator &,
  const dealii::Triangulation<2, 2>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<1, 2> &,
  const dealii::Mapping<2, 2> &);


template dealii::Quadrature<2>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<2, 2>::cell_iterator &,
  const dealii::Triangulation<2, 2>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<2, 2> &,
  const dealii::Mapping<2, 2> &);


template dealii::Quadrature<3>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<2, 3>::cell_iterator &,
  const dealii::Triangulation<3, 3>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<2, 3> &,
  const dealii::Mapping<3, 3> &);

template dealii::Quadrature<3>
dealii::NonMatching::compute_intersection(
  const dealii::Triangulation<3, 3>::cell_iterator &,
  const dealii::Triangulation<3, 3>::cell_iterator &,
  const unsigned int,
  const dealii::Mapping<3, 3> &,
  const dealii::Mapping<3, 3> &);



template std::vector<
  std::tuple<typename dealii::Triangulation<2, 2>::active_cell_iterator,
             typename dealii::Triangulation<1, 2>::active_cell_iterator,
             Quadrature<2>>>
NonMatching::compute_intersection(const GridTools::Cache<2, 2> &space_cache,
                                  const GridTools::Cache<1, 2> &immersed_cache,
                                  const unsigned int            degree);

template std::vector<
  std::tuple<typename dealii::Triangulation<2, 2>::active_cell_iterator,
             typename dealii::Triangulation<2, 2>::active_cell_iterator,
             Quadrature<2>>>
NonMatching::compute_intersection(const GridTools::Cache<2, 2> &space_cache,
                                  const GridTools::Cache<2, 2> &immersed_cache,
                                  const unsigned int            degree);

template std::vector<
  std::tuple<typename dealii::Triangulation<3, 3>::active_cell_iterator,
             typename dealii::Triangulation<2, 3>::active_cell_iterator,
             Quadrature<3>>>
NonMatching::compute_intersection(const GridTools::Cache<3, 3> &space_cache,
                                  const GridTools::Cache<2, 3> &immersed_cache,
                                  const unsigned int            degree);


template std::vector<
  std::tuple<typename dealii::Triangulation<3, 3>::active_cell_iterator,
             typename dealii::Triangulation<3, 3>::active_cell_iterator,
             Quadrature<3>>>
NonMatching::compute_intersection(const GridTools::Cache<3, 3> &space_cache,
                                  const GridTools::Cache<3, 3> &immersed_cache,
                                  const unsigned int            degree);

#endif