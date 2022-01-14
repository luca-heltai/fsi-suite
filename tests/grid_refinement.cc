#include "parsed_tools/grid_refinement.h"

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTesterNoOne, GridRefinementGlobal)
{
  Triangulation<this->dim, this->spacedim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  ParsedTools::GridRefinement pgr("/");
  ParameterAcceptor::initialize();

  Vector<float> criteria(tria.n_active_cells());
  criteria = 1.0;
  pgr.mark_cells(criteria, tria);
  tria.execute_coarsening_and_refinement();

  ASSERT_EQ(tria.n_active_cells(), std::pow(2, (this->dim * 2)));
}