#include "parsed_grid_generator.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "tests.h"

using namespace dealii;

using DimSpacedimTypes = ::testing::Types<
  std::tuple<std::integral_constant<int, 1>, std::integral_constant<int, 1>>,
  std::tuple<std::integral_constant<int, 1>, std::integral_constant<int, 2>>,
  std::tuple<std::integral_constant<int, 1>, std::integral_constant<int, 3>>,
  std::tuple<std::integral_constant<int, 2>, std::integral_constant<int, 2>>,
  std::tuple<std::integral_constant<int, 2>, std::integral_constant<int, 3>>,
  std::tuple<std::integral_constant<int, 3>, std::integral_constant<int, 3>>>;

template <class DimSpacedim>
class PGGTester : public ::testing::Test
{
public:
  PGGTester()
    : pgg("/"){};

  using Tdim      = typename std::tuple_element<0, DimSpacedim>::type;
  using Tspacedim = typename std::tuple_element<1, DimSpacedim>::type;

  const unsigned int dim      = Tdim::value;
  const unsigned int spacedim = Tspacedim::value;

  ParsedGridGenerator<Tdim::value, Tspacedim::value> pgg;

  Triangulation<Tdim::value, Tspacedim::value> tria;

  void
  parse(const std::string &prm_string) const
  {
    ParameterAcceptor::prm.parse_input_from_string(prm_string);
    ParameterAcceptor::parse_all_parameters();
  }
};

TYPED_TEST_CASE(PGGTester, DimSpacedimTypes);

TYPED_TEST(PGGTester, GenerateHyperCube)
{
  this->parse(R"(
    set Input name = hyper_cube
    set Arguments = 0: 1: false
    set Output name = grid.msh
    set Transform to simplex grid = false
  )");

  // After this, we should have a file grid.msh
  this->pgg.generate(this->tria);
  ASSERT_TRUE(std::ifstream("grid.msh"));
  std::remove("grid.msh");

  // And the grid should have 1 element
  ASSERT_EQ(this->tria.n_active_cells(), 1u);
}


TYPED_TEST(PGGTester, DISABLED_GenerateHyperCubeSimplices)
{
  this->parse(R"(
    set Input name = hyper_cube
    set Arguments = 0: 1: false
    set Output name = grid.msh
    set Transform to simplex grid = true
  )");

  // After this, we should have a file grid.msh
  this->pgg.generate(this->tria);
  ASSERT_TRUE(std::ifstream("grid.msh"));
  std::remove("grid.msh");

  // And the grid should have 8 elements in 2d, and 24 in 3d
  const unsigned int dims[] = {0, 1, 8, 24};
  ASSERT_EQ(this->tria.n_active_cells(), dims[this->dim]);
}
