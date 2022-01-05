#include <deal.II/base/config.h>

#include "tools/components.h"

#include <deal.II/base/patterns.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
using namespace dealii;

TEST(Components, SimpleComponentsChecks)
{
  const std::string names = "u, u, p";
  ASSERT_EQ(Tools::Components::n_components(names), 3u);
  ASSERT_EQ(Tools::Components::n_blocks(names), 2u);

  const auto &[b, m] = Tools::Components::names_to_blocks(names);
  ASSERT_EQ(b.size(), 2u);
  ASSERT_EQ(m.size(), 2u);
  ASSERT_EQ(b[0], "u");
  ASSERT_EQ(m[0], 2u);
  ASSERT_EQ(b[1], "p");
  ASSERT_EQ(m[1], 1u);

  ASSERT_EQ(Tools::Components::blocks_to_names(b, m), names);

  // Now test that masks work correctly.
  auto mask = Tools::Components::mask(names, "u");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, false");

  mask = Tools::Components::mask(names, "p");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "false, false, true");

  mask = Tools::Components::mask(names, "u.N");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, false");

  mask = Tools::Components::mask(names, "u.n");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, false");

  mask = Tools::Components::mask(names, "0, 1");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, false");

  mask = Tools::Components::mask(names, "0");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, false, false");

  mask = Tools::Components::mask(names, "1, 2");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "false, true, true");

  mask = Tools::Components::mask(names, "ALL");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, true");

  mask = Tools::Components::mask(names, "all");
  ASSERT_EQ(Patterns::Tools::to_string(mask), "true, true, true");

  ASSERT_ANY_THROW(Tools::Components::mask(names, "u.T"));
  ASSERT_ANY_THROW(Tools::Components::mask(names, "v"));
  ASSERT_ANY_THROW(Tools::Components::mask(names, "3"));
}
