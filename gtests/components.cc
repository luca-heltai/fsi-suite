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

#include "parsed_tools/components.h"

#include <deal.II/base/patterns.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
using namespace dealii;
using namespace ParsedTools::Components;



TEST(Components, NamesToBlocks)
{
  const std::string names = "u, u, p";
  ASSERT_EQ(n_components(names), 3u);
  ASSERT_EQ(n_blocks(names), 2u);

  const auto &[b, m] = names_to_blocks(names);
  ASSERT_EQ(b.size(), 2u);
  ASSERT_EQ(m.size(), 2u);
  ASSERT_EQ(b[0], "u");
  ASSERT_EQ(m[0], 2u);
  ASSERT_EQ(b[1], "p");
  ASSERT_EQ(m[1], 1u);

  ASSERT_EQ(blocks_to_names(b, m), names);
}



TEST(Components, BlockIndices)
{
  const std::string names = "u, u, p";

  auto r = block_indices(names, "u");
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = block_indices(names, "u, p"); // {0,1
  ASSERT_EQ(r.size(), 2u);
  ASSERT_EQ(r[0], 0u);
  ASSERT_EQ(r[1], 1u);

  r = block_indices(names, "p"); // {1}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 1u);

  r = block_indices(names, "2"); // {1}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 1u);

  r = block_indices(names, "1"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = block_indices(names, "0,1,2"); // {0,0,1}
  ASSERT_EQ(r.size(), 3u);
  ASSERT_EQ(r[0], 0u);
  ASSERT_EQ(r[1], 0u);
  ASSERT_EQ(r[2], 1u);

  r = block_indices(names, "u.n"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = block_indices(names, "u.t"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);
}



TEST(Components, ComponentIndices)
{
  const std::string names = "u, u, p";

  auto r = component_indices(names, "u");
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = component_indices(names, "u, p"); // {0,2}
  ASSERT_EQ(r.size(), 2u);
  ASSERT_EQ(r[0], 0u);
  ASSERT_EQ(r[1], 2u);

  r = component_indices(names, "p"); // {2}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 2u);

  r = component_indices(names, "2"); // {2}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 2u);

  r = component_indices(names, "1"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = component_indices(names, "0,1,2"); // {0,0,2}
  ASSERT_EQ(r.size(), 3u);
  ASSERT_EQ(r[0], 0u);
  ASSERT_EQ(r[1], 0u);
  ASSERT_EQ(r[2], 2u);

  r = component_indices(names, "u.n"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);

  r = component_indices(names, "u.t"); // {0}
  ASSERT_EQ(r.size(), 1u);
  ASSERT_EQ(r[0], 0u);
}



TEST(Components, Mask)
{
  const std::string names = "u, u, p";

  auto m = mask(names, "u");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, false");

  m = mask(names, "p");
  ASSERT_EQ(Patterns::Tools::to_string(m), "false, false, true");

  m = mask(names, "u.N");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, false");

  m = mask(names, "u.n");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, false");

  m = mask(names, "0, 1");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, false");

  m = mask(names, "0");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, false, false");

  m = mask(names, "1, 2");
  ASSERT_EQ(Patterns::Tools::to_string(m), "false, true, true");

  m = mask(names, "ALL");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, true");

  m = mask(names, "all");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, true");

  m = mask(names, "u.T");
  ASSERT_EQ(Patterns::Tools::to_string(m), "true, true, false");

  ASSERT_ANY_THROW(mask(names, "v"));
  ASSERT_ANY_THROW(mask(names, "3"));
}



TEST(Components, Type)
{
  const std::string names = "u, u, p";

  auto t = type(names, "u");
  ASSERT_EQ(t, Type::vector);

  t = type(names, "u.n");
  ASSERT_EQ(t, Type::normal);

  t = type(names, "u.N");
  ASSERT_EQ(t, Type::normal);

  t = type(names, "u.t");
  ASSERT_EQ(t, Type::tangential);

  t = type(names, "u.T");
  ASSERT_EQ(t, Type::tangential);

  t = type(names, "all");
  ASSERT_EQ(t, Type::all);

  t = type(names, "ALL");
  ASSERT_EQ(t, Type::all);

  t = type(names, "1");
  ASSERT_EQ(t, Type::vector);

  t = type(names, "0");
  ASSERT_EQ(t, Type::vector);

  t = type(names, "2");
  ASSERT_EQ(t, Type::scalar);

  ASSERT_ANY_THROW(type(names, "v"));
  ASSERT_ANY_THROW(type(names, "3"));
}