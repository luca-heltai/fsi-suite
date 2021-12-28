#include "tools/parsed_grid_refinement.h"

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DSTNoOne, ParsedGridRefinementGlobal)
{
  Triangulation<this->dim, this->spacedim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  Tools::ParsedGridRefinement pgr("/", "global", .3, 0.0, 0, 0, 0);
  ParameterAcceptor::initialize();

  Vector<float> criteria(tria.n_active_cells());
  criteria = 1.0;
  pgr.mark_cells(criteria, tria);
  tria.execute_coarsening_and_refinement();

  ASSERT_EQ(tria.n_active_cells(), std::pow(2, (this->dim * 2)));
}