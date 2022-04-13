#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <deal.II/base/point.h>
#  include <deal.II/base/quadrature.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/reference_cell.h>
#  include <deal.II/grid/tria.h>

#  include <moonolith_build_quadrature.hpp>
#  include <moonolith_map_quadrature.hpp>
#  include <moonolith_polygon.hpp>
#  include <moonolith_vector.hpp>

#  include <array>
#  include <cassert>
#  include <cmath>
#  include <fstream>
#  include <iostream>
#  include <sstream>

#  include "dim_spacedim_tester.h"
using namespace dealii;
using namespace moonolith;


template <int dim, typename NumberType>
inline dealii::Point<dim, NumberType>
to_dealii(const moonolith::Vector<NumberType, dim> &p)
{
  dealii::Point<dim, NumberType> result;
  for (unsigned int i = 0; i < dim; ++i)
    result[i] = p[i];
  return result;
}



template <int dim, typename NumberType>
inline moonolith::Vector<NumberType, dim>
to_moonolith(const dealii::Point<dim, NumberType> &p)
{
  moonolith::Vector<NumberType, dim> result;
  for (unsigned int i = 0; i < dim; ++i)
    result[i] = p[i];
  return result;
}



template <int dim>
inline dealii::Quadrature<dim>
to_dealii(const moonolith::Quadrature<double, dim> &q)
{
  const auto                      points  = q.get_points();
  const auto                      weights = q.get_weights();
  std::vector<dealii::Point<dim>> dealii_points(points.size());
  for (unsigned int i = 0; i < points.size(); ++i)
    dealii_points[i] = to_dealii(points[i]);
  return dealii::Quadrature<dim>(dealii_points, weights);
}



template <int dim>
Line<double, dim>
to_moonolith(const typename Triangulation<1, dim>::cell_iterator &cell,
             const Mapping<1, dim> &                              mapping)
{
  const auto        vertices = mapping.get_vertices(cell);
  Line<double, dim> line;
  line.p0 = to_moonolith(vertices[0]);
  line.p1 = to_moonolith(vertices[1]);
  return line;
}



template <int spacedim>
Polygon<double, spacedim>
to_moonolith(const typename Triangulation<2, spacedim>::cell_iterator &cell,
             const Mapping<2, spacedim> &                              mapping)
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


TYPED_TEST(DimTester, CheckMoonoLithTranslations)
{
  constexpr auto dim = TestFixture::dim;

  dealii::Point<dim> p;
  for (unsigned int i = 0; i < dim; ++i)
    p[i] = i;

  const auto q = to_moonolith(p);

  ASSERT_EQ(length(q), p.norm());
}


TYPED_TEST(DimTester, MoonolithLines)
{
  constexpr auto dim = TestFixture::dim;


  Triangulation<1, dim> tria;
  const auto            ref     = ReferenceCell::n_vertices_to_type(1, 2);
  const auto            mapping = ref.template get_default_mapping<1, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  ASSERT_NEAR(measure(poly), 1.0, 1e-10);
}

#endif