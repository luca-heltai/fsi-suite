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

#ifndef moonolith_tools_include_h
#define moonolith_tools_include_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <deal.II/base/point.h>
#  include <deal.II/base/quadrature.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/reference_cell.h>
#  include <deal.II/grid/tria.h>

#  include <moonolith_build_quadrature.hpp>
#  include <moonolith_convex_decomposition.hpp>
#  include <moonolith_intersect_polyhedra.hpp>
#  include <moonolith_map_quadrature.hpp>
#  include <moonolith_mesh_io.hpp>
#  include <moonolith_par_l2_transfer.hpp>
#  include <moonolith_polygon.hpp>
#  include <moonolith_vector.hpp>
#  include <par_moonolith.hpp>

#  include <array>
#  include <cassert>
#  include <cmath>
#  include <fstream>
#  include <iostream>
#  include <sstream>

namespace moonolith
{
  /**
   * Convert a moonolith::Vector with dimension dim to a dealii::Point<dim>.
   */
  using namespace dealii;
  template <int dim, typename NumberType>
  inline dealii::Point<dim, NumberType>
  to_dealii(const moonolith::Vector<NumberType, dim> &p)
  {
    dealii::Point<dim, NumberType> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = p[i];
    return result;
  }

  /**
   * Convert a dealii::Point<dim> to a moonolith::Vector.
   */
  template <int dim, typename NumberType>
  inline moonolith::Vector<NumberType, dim>
  to_moonolith(const dealii::Point<dim, NumberType> &p)
  {
    moonolith::Vector<NumberType, dim> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = p[i];
    return result;
  }


  /**
   * Convert a moonolith::Quadrature<dim> to a dealii::Quadrature<dim>.
   */
  template <int dim>
  inline dealii::Quadrature<dim>
  to_dealii(const moonolith::Quadrature<double, dim> &q)
  {
    const auto                      points    = q.points;
    const auto                      weights   = q.weights;
    const unsigned int              quad_size = points.size();
    std::vector<dealii::Point<dim>> dealii_points(quad_size);
    std::vector<double>             dealii_weights(quad_size);
    for (unsigned int i = 0; i < quad_size; ++i)
      {
        dealii_points[i]  = to_dealii(points[i]);
        dealii_weights[i] = weights[i];
      }
    return dealii::Quadrature<dim>(dealii_points, dealii_weights);
  }


  /**
   * Convert a 1 dimensional cell to a moonolith::Line.
   */
  template <int dim>
  inline Line<double, dim>
  to_moonolith(const typename Triangulation<1, dim>::cell_iterator &cell,
               const Mapping<1, dim> &                              mapping)
  {
    const auto        vertices = mapping.get_vertices(cell);
    Line<double, dim> line;
    line.p0 = to_moonolith(vertices[0]);
    line.p1 = to_moonolith(vertices[1]);
    return line;
  }


  /**
   * Convert a 2 dimensional cell to a moonolith::Polygon.
   */
  template <int spacedim>
  inline Polygon<double, spacedim>
  to_moonolith(const typename Triangulation<2, spacedim>::cell_iterator &cell,
               const Mapping<2, spacedim> &mapping)
  {
    static_assert(2 <= spacedim, "2 must be <= spacedim");
    const auto                vertices = mapping.get_vertices(cell);
    Polygon<double, spacedim> poly;

    if (vertices.size() == 3)
      poly.points = {to_moonolith(vertices[0]),
                     to_moonolith(vertices[1]),
                     to_moonolith(vertices[2])};
    else if (vertices.size() == 4)
      poly.points = {to_moonolith(vertices[0]),
                     to_moonolith(vertices[1]),
                     to_moonolith(vertices[3]),
                     to_moonolith(vertices[2])};
    else
      Assert(false, ExcNotImplemented());

    return poly;
  }

  /**
   * Convert a three dimensional cell to a moonolith::Polyedron.
   */
  inline Polyhedron<double>
  to_moonolith(const typename Triangulation<3, 3>::cell_iterator &cell,
               const Mapping<3, 3> &                              mapping)
  {
    const auto         vertices = mapping.get_vertices(cell);
    Polyhedron<double> poly;

    if (vertices.size() == 4) // Tetrahedron
      {
        poly.el_index = {0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3};
        poly.el_ptr   = {0, 3, 6, 9, 12};

        poly.points = {to_moonolith(vertices[0]),
                       to_moonolith(vertices[1]),
                       to_moonolith(vertices[2]),
                       to_moonolith(vertices[3])};

        poly.fix_ordering();
      }
    else if (vertices.size() == 8) // Hexahedron
      {
        make_cube(to_moonolith(vertices[0]), to_moonolith(vertices[7]), poly);
        poly.points = {to_moonolith(vertices[0]),
                       to_moonolith(vertices[1]),
                       to_moonolith(vertices[3]),
                       to_moonolith(vertices[2]),
                       to_moonolith(vertices[4]),
                       to_moonolith(vertices[5]),
                       to_moonolith(vertices[7]),
                       to_moonolith(vertices[6])};
        poly.fix_ordering();
      }
    else if (vertices.size() == 5) // Piramid
      {
        poly.el_index = {0, 1, 3, 2, 0, 1, 4, 1, 3, 4, 3, 4, 2, 0, 4, 2};
        poly.el_ptr   = {0, 4, 7, 10, 13, 16};

        poly.points = {to_moonolith(vertices[0]),
                       to_moonolith(vertices[1]),
                       to_moonolith(vertices[2]),
                       to_moonolith(vertices[3]),
                       to_moonolith(vertices[4])};

        poly.fix_ordering();
      }
    else if (vertices.size() == 6) // Wedge
      {
        poly.el_index = {0, 1, 2, 0, 1, 4, 3, 1, 4, 5, 2, 0, 2, 5, 3, 3, 4, 5};
        poly.el_ptr   = {0, 3, 7, 11, 15, 18};

        poly.points = {to_moonolith(vertices[0]),
                       to_moonolith(vertices[1]),
                       to_moonolith(vertices[2]),
                       to_moonolith(vertices[3]),
                       to_moonolith(vertices[4]),
                       to_moonolith(vertices[5])};

        poly.fix_ordering();
      }
    return poly;
  }

  /**
   * Compute the intersection beetween two arbitrary shapes using the moonolith
   * library.
   *
   * @tparam spacedim The dimension of the embedding space
   * @tparam dim The minimum of the intrinsic dimensions of the two shapes
   * @tparam T1 First shape type
   * @tparam T2 Second shape type
   * @param ref_quad A reference quadrature on simplices of dimension dim, used
   * to integrate on a triangulation of the intersection
   * @param t1 First shape
   * @param t2 Second shape
   * @return moonolith::Quadrature<double, spacedim>
   */
  template <int spacedim, int dim, class T1, class T2>
  inline moonolith::Quadrature<double, spacedim>
  compute_intersection(const moonolith::Quadrature<double, dim> &ref_quad,
                       const T1 &                                t1,
                       const T2 &                                t2)
  {
    BuildQuadrature<T1, T2>      intersect;
    Quadrature<double, spacedim> out;
    intersect.apply(ref_quad, t1, t2, out);
    return out;
  }


  /**
   * Compute the intersection beetween two arbitrary deal.II cells using the
   * moonolith library, and return a quadrature formula which can be used to
   * integrate exactly on the intersection.
   *
   * @tparam spacedim The dimension of the embedding space
   * @tparam dim0 intrinsic dimension of the first shape
   * @tparam dim1 intrinsic dimension of the second shape
   * @tparam T1 First shape type
   * @tparam T2 Second shape type
   * @param cell0 First chell
   * @param cell1 Second cell
   * @param order Order of the reference quadrature on simplices of dimension
   * min(dim0,dim1), used to integrate on a triangulation of the intersection
   * @return Quadrature<spacedim>
   */
  template <int dim0, int dim1, int spacedim>
  inline dealii::Quadrature<spacedim>
  compute_intersection(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                           degree,
    const Mapping<dim0, spacedim> &                              mapping0,
    const Mapping<dim1, spacedim> &                              mapping1)
  {
    const auto t0 = to_moonolith(cell0, mapping0);
    const auto t1 = to_moonolith(cell1, mapping1);
    moonolith::Quadrature<Real, std::min(dim0, dim1)> ref_quad;
    moonolith::Gauss::get(degree, ref_quad);
    auto mquad = compute_intersection<spacedim>(ref_quad, t0, t1);
    return to_dealii(mquad);
  }
} // namespace moonolith

#endif
#endif