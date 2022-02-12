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

#include "parsed_tools/enum.h"

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/fe_values.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace dealii;

TEST(ParsedEnum, CheckFEValuesFlags)
{
  UpdateFlags flags;

  ParameterAcceptor::prm.add_parameter("Update flags", flags);
  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Update flags = update_values
  )");

  ASSERT_TRUE(flags & update_values)
    << Patterns::Tools::Convert<UpdateFlags>::to_string(flags);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Update flags = update_values | update_gradients
  )");

  auto s = Patterns::Tools::Convert<UpdateFlags>::to_string(flags);
  ASSERT_EQ(s, "update_default| update_values| update_gradients");
  ASSERT_TRUE(flags & (update_values | update_gradients));
}