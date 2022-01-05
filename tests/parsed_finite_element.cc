#include "tools/parsed_finite_element.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTester, ParsedFiniteElementScalar)
{
  Tools::ParsedFiniteElement<this->dim, this->spacedim> pfe("/fe" + this->id(),
                                                            "u",
                                                            "FE_Q(1)");

  this->parse(R"(
    set Finite element space (u) = FE_Q(1)
  )",
              pfe);

  FiniteElement<this->dim, this->spacedim> &fe = pfe;

  ASSERT_EQ(fe.n_components(), 1u);
  ASSERT_EQ(fe.n_blocks(), 1u);
  ASSERT_EQ(fe.n_base_elements(), 1u);
  ASSERT_EQ(fe.dofs_per_cell, std::pow(2, this->dim));
}