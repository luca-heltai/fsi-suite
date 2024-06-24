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

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <moonolith_tools.h>

#  include "dim_spacedim_tester.h"

using namespace dealii;
using namespace moonolith;


TYPED_TEST(DimTester, CheckMoonoLithConversions)
{
  constexpr unsigned int dim = TestFixture::dim;

  dealii::Point<dim> p;
  for (unsigned int i = 0; i < dim; ++i)
    p[i] = i;

  const auto q = to_moonolith(p);

  ASSERT_EQ(length(q), p.norm());
}



TYPED_TEST(DimTester, CheckMoonoLithQuadConversions)
{
  constexpr unsigned int dim = TestFixture::dim;

  moonolith::Quadrature<Real, dim> ref_quad;
  const auto                       dealii_quad = to_dealii(ref_quad);
  const auto                       pts         = dealii_quad.get_points();
  const auto                       wts         = dealii_quad.get_weights();
  for (unsigned int i = 0; i < dealii_quad.size(); ++i)
    {
      ASSERT_EQ(pts[i], to_dealii(ref_quad.points[i]));
      ASSERT_EQ(wts[i], ref_quad.weights[i]);
    }
}



TYPED_TEST(DimTester, MoonolithLines)
{
  constexpr unsigned int dim = TestFixture::dim;


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
  constexpr unsigned int dim = 3; // TestFixture::dim;

  Triangulation<3> tria;
  const auto       ref     = ReferenceCell::n_vertices_to_type(3, 4);
  const auto       mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  ASSERT_NEAR(measure(poly), 0.16666666666666666, 1e-10);
}


TEST(Moonolith3, MoonolithExa)
{
  constexpr unsigned int dim = 3; // TestFixture::dim;

  Triangulation<3> tria;
  const auto       ref     = ReferenceCell::n_vertices_to_type(3, 8);
  const auto       mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  // MatlabScripter plot;
  // plot.plot(poly);
  // plot.save("cubo.m");

  ASSERT_NEAR(measure(poly), 1, 1e-10);
}


TEST(Moonolith3, MoonolithPiramid)
{
  constexpr unsigned int dim = 3; // TestFixture::dim;

  Triangulation<3> tria;
  const auto       ref     = ReferenceCell::n_vertices_to_type(3, 5);
  const auto       mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  // MatlabScripter plot;
  // plot.plot(poly);
  // plot.save("piramide.m");

  ASSERT_NEAR(measure(poly), 1.3333333333333333, 1e-10);
}


TEST(Moonolith3, MoonolithWedge)
{
  constexpr unsigned int dim = 3; // TestFixture::dim;

  Triangulation<3> tria;
  const auto       ref     = ReferenceCell::n_vertices_to_type(3, 6);
  const auto       mapping = ref.template get_default_mapping<3, dim>(1);

  GridGenerator::reference_cell(tria, ref);
  const auto cell = tria.begin_active();
  const auto poly = to_moonolith(cell, *mapping);

  // MatlabScripter plot;
  // plot.plot(poly);
  // plot.save("wedge.m");

  ASSERT_NEAR(measure(poly), 0.5, 1e-10);
}


TYPED_TEST(DimSpacedimTester, AllMoonolithConversions)
{
  constexpr unsigned int dim      = TestFixture::dim;
  constexpr auto         spacedim = TestFixture::spacedim;

  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};
  std::vector<std::vector<double>>       measures = {{},
                                                     {1},
                                                     {0.5, 1.0},
                                                     {1. / 6., 4. / 3., 0.5, 1}};

  unsigned int i = 0;
  for (const auto nv : d2t[dim])
    {
      Triangulation<dim, spacedim> tria;
      const auto ref     = ReferenceCell::n_vertices_to_type(dim, nv);
      const auto mapping = ref.template get_default_mapping<dim, spacedim>(1);

      GridGenerator::reference_cell(tria, ref);

      const auto cell = tria.begin_active();
      const auto poly = to_moonolith(cell, *mapping);

      ASSERT_NEAR(measure(poly), measures[dim][i++], 1e-10);
    }
}


TYPED_TEST(DimSpacedimTesterNoOne, AllMoonolithConversionsRotated)
{
  constexpr unsigned int dim      = TestFixture::dim;
  constexpr auto         spacedim = TestFixture::spacedim;

  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};
  std::vector<std::vector<double>>       measures = {{},
                                                     {1},
                                                     {0.5, 1.0},
                                                     {1. / 6., 4. / 3., 0.5, 1}};

  unsigned int i = 0;
  for (const auto nv : d2t[dim])
    {
      Triangulation<dim, spacedim> tria;
      const auto ref     = ReferenceCell::n_vertices_to_type(dim, nv);
      const auto mapping = ref.template get_default_mapping<dim, spacedim>(1);

      GridGenerator::reference_cell(tria, ref);
      if constexpr (spacedim == 2 && dim == 2)
        {
          GridTools::rotate(numbers::PI_4, tria);
        }
      else if constexpr (spacedim == 3)
        {
          GridTools::rotate(Tensor<1, 3>({1., 0., 0.}), numbers::PI_4, tria);
        }

      const auto cell = tria.begin_active();
      const auto poly = to_moonolith(cell, *mapping);

      ASSERT_NEAR(measure(poly), measures[dim][i++], 1e-10);
    }
}
#endif