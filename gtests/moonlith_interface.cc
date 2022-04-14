#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_PARMOONOLITH

#  include <deal_moon_conversion.h>

#  include "dim_spacedim_tester.h"

using namespace dealii;
using namespace moonolith;


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
  constexpr auto dim = 3; // TestFixture::dim;

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
  constexpr auto dim = 3; // TestFixture::dim;

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
  constexpr auto dim = 3; // TestFixture::dim;

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
  constexpr auto dim = 3; // TestFixture::dim;

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
  constexpr auto dim      = TestFixture::dim;
  constexpr auto spacedim = TestFixture::spacedim;

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
  constexpr auto dim      = TestFixture::dim;
  constexpr auto spacedim = TestFixture::spacedim;

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
          GridTools::rotate(numbers::PI_4, 0, tria);
        }

      const auto cell = tria.begin_active();
      const auto poly = to_moonolith(cell, *mapping);

      ASSERT_NEAR(measure(poly), measures[dim][i++], 1e-10);
    }
}
#endif