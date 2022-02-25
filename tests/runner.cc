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

#include "runner.h"

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace dealii;

TEST(Runner, DimAndSpacedim)
{
  {
    // use -d=2 to specify flag/value
    char *argv[] = {(char *)"./app", (char *)"-d=2", (char *)"-s=3", NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 2);
    EXPECT_EQ(spacedim, 3);
    EXPECT_EQ(in_file, "");
    EXPECT_EQ(out_file, "used_app_2d_3d.prm");
  }
  {
    // use -d=1 to specify flag value, check default value for spacedim
    char *argv[] = {(char *)"./app", (char *)"-d=1", NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 1);
    EXPECT_EQ(spacedim, 1);
    EXPECT_EQ(in_file, "");
    EXPECT_EQ(out_file, "used_app_1d.prm");
  }
  {
    // use -d 3 to specify flag value, check default value for spacedim
    char *argv[] = {(char *)"./app", (char *)"-d", (char *)"3", NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 3);
    EXPECT_EQ(spacedim, 3);
    EXPECT_EQ(in_file, "");
    EXPECT_EQ(out_file, "used_app_3d.prm");
  }
}


TEST(Runner, PrmName)
{
  {
    // default dimension and spacedim, with input file
    char *argv[] = {(char *)"./app", (char *)"-i", (char *)"input.prm", NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 2);
    EXPECT_EQ(spacedim, 2);
    EXPECT_EQ(in_file, "input.prm");
    EXPECT_EQ(out_file, "used_input.prm");
  }
  {
    // use filename for dim
    char *argv[] = {(char *)"./app",
                    (char *)"-i",
                    (char *)"input_3d.prm",
                    NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 3);
    EXPECT_EQ(spacedim, 3);
    EXPECT_EQ(in_file, "input_3d.prm");
    EXPECT_EQ(out_file, "used_input_3d.prm");
  }
  {
    // use filename for dim and spacedim
    char *argv[] = {(char *)"./app",
                    (char *)"-i",
                    (char *)"input_1d_3d.prm",
                    NULL};
    auto [dim, spacedim, in_file, out_file] =
      Runner::get_dimensions_and_parameter_files(argv);
    EXPECT_EQ(dim, 1);
    EXPECT_EQ(spacedim, 3);
    EXPECT_EQ(in_file, "input_1d_3d.prm");
    EXPECT_EQ(out_file, "used_input_1d_3d.prm");
  }
  {
    // check failure if file name and dimensions do not correspond
    char *argv[] = {(char *)"./app",
                    (char *)"-d",
                    (char *)"2",
                    (char *)"-i",
                    (char *)"input_1d_3d.prm",
                    NULL};
    EXPECT_ANY_THROW(Runner::get_dimensions_and_parameter_files(argv));
  }
}



TEST(Runner, PrmFromCommandLine)
{
  // check that we know how to parse a parameter
  unsigned int verbosity = 0;
  ParameterAcceptor::prm.add_parameter("verbosity", verbosity);
  char *argv[] = {(char *)"./app",
                  (char *)"-verbosity",
                  (char *)"1",
                  (char *)"-o",
                  (char *)"output.prm",
                  NULL};

  auto [dim, spacedim, in_file, out_file] =
    Runner::get_dimensions_and_parameter_files(argv);
  Runner::setup_parameters_from_cli(argv, in_file, out_file);
  EXPECT_EQ(verbosity, 1);
  ASSERT_TRUE(std::ifstream("output.prm"));
  std::remove("output.prm");
}
