#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <deal.II/base/point.h>
#  include <deal.II/base/quadrature.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/reference_cell.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/grid_tools.h>

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

template <int spacedim>
Polyhedron<double>
to_moonolith(const typename Triangulation<3,spacedim>::cell_iterator &cell,
             const Mapping<3, spacedim> &                              mapping)
{
  static_assert(3 <= spacedim, "3 must be <= spacedim");
  const auto                vertices = mapping.get_vertices(cell);
  Polyhedron<double> poly;

  if (vertices.size() == 4) // Tetrahedron
  {
    poly.el_index = {0, 2, 1,
                     0, 3, 2, 
                     0, 1, 3, 
                     1, 2, 3};
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
    poly.el_index = {0, 1, 3, 2,
                     0, 1, 4,
                     1, 3, 4,
                     3, 4, 2,
                     0, 4, 2};
    poly.el_ptr   = {0,4,7,10,13,16};

    poly.points = {to_moonolith(vertices[0]),
                   to_moonolith(vertices[1]),
                   to_moonolith(vertices[2]),
                   to_moonolith(vertices[3]),
                   to_moonolith(vertices[4])};

    poly.fix_ordering();
  }
  else if (vertices.size() == 6) // Wedge
  {
    poly.el_index = {0, 1, 2,
                     0, 1, 4, 3,
                     1, 4, 5, 2,
                     0, 2, 5, 3,
                     3, 4, 5};
    poly.el_ptr = {0,3,7,11,15,18};

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


TYPED_TEST(DimTester, CheckMoonoLithConversions)
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


TEST(Moonolith3, MoonolithTetra)
{
  constexpr auto dim = 3; //TestFixture::dim;

  Triangulation<3>      tria;
  const auto            ref     = ReferenceCell::n_vertices_to_type(3, 4);
  const auto            mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  ASSERT_NEAR(measure(poly), 0.16666666666666666, 1e-10);
}


TEST(Moonolith3, MoonolithExa)
{
  constexpr auto dim = 3; //TestFixture::dim;

  Triangulation<3>      tria;
  const auto            ref     = ReferenceCell::n_vertices_to_type(3, 8);
  const auto            mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  //MatlabScripter plot;
  //plot.plot(poly);
  //plot.save("cubo.m");

  ASSERT_NEAR(measure(poly), 1, 1e-10);
}


TEST(Moonolith3, MoonolithPiramid)
{
  constexpr auto dim = 3; //TestFixture::dim;

  Triangulation<3>      tria;
  const auto            ref     = ReferenceCell::n_vertices_to_type(3, 5);
  const auto            mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  //MatlabScripter plot;
  //plot.plot(poly);
  //plot.save("piramide.m");

  ASSERT_NEAR(measure(poly), 1.3333333333333333, 1e-10);
}


TEST(Moonolith3, MoonolithWedge)
{
  constexpr auto dim = 3; //TestFixture::dim;

  Triangulation<3>      tria;
  const auto            ref     = ReferenceCell::n_vertices_to_type(3, 6);
  const auto            mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  //MatlabScripter plot;
  //plot.plot(poly);
  //plot.save("wedge.m");

  ASSERT_NEAR(measure(poly), 0.5, 1e-10);
}


TYPED_TEST(DimSpacedimTester, AllMoonolithConversions)
{
  constexpr auto dim = TestFixture::dim;
  constexpr auto spacedim = TestFixture::spacedim;

  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};
  std::vector<std::vector<double>> measures = {{}, {1}, {0.5, 1.0}, {1./6.,4./3., 0.5, 1}};

  unsigned int i = 0;
  for(const auto nv:d2t[dim]){
    Triangulation<dim,spacedim>      tria;
    const auto            ref     = ReferenceCell::n_vertices_to_type(dim, nv);
    const auto            mapping = ref.template get_default_mapping<dim, spacedim>(1);

    GridGenerator::reference_cell(tria, ref);

    const auto cell = tria.begin_active();
    const auto poly = to_moonolith(cell, *mapping);

    ASSERT_NEAR(measure(poly), measures[dim][i++], 1e-10);
  }
}


TYPED_TEST(DimSpacedimTesterNoOne, AllMoonolithConversionsRotated)
{
  constexpr auto dim = TestFixture::dim;
  constexpr auto spacedim = TestFixture::spacedim;

  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};
  std::vector<std::vector<double>> measures = {{}, {1}, {0.5, 1.0}, {1./6.,4./3., 0.5, 1}};

  unsigned int i = 0;
  for(const auto nv:d2t[dim]){
    Triangulation<dim,spacedim>      tria;
    const auto            ref     = ReferenceCell::n_vertices_to_type(dim, nv);
    const auto            mapping = ref.template get_default_mapping<dim, spacedim>(1);

    GridGenerator::reference_cell(tria, ref);
    if constexpr (spacedim == 2 && dim == 2){
      GridTools::rotate(numbers::PI_4,tria);
    }
    else if constexpr (spacedim == 3){
      GridTools::rotate(numbers::PI_4,0,tria);
    }

    const auto cell = tria.begin_active();
    const auto poly = to_moonolith(cell, *mapping);

    ASSERT_NEAR(measure(poly), measures[dim][i++], 1e-10);
  }
}
#endif