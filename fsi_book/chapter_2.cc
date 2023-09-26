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

#include <deal.II/grid/manifold_lib.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "pdes/mpi/poisson.h"

using namespace dealii;

TEST(Chapter2, Example2_1_Matching_Interface)
{
  static const int        dim = 2;
  PDEs::MPI::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_2_example_2_1_matching_balls.prm",
    "chapter_2_example_2_1_matching_balls.prm");

  // Make a circle inside a square
  poisson.grid_generator_call_back.connect([](auto &triangulation) {
    const double inner_radius = .25;
    const double outer_radius = .5;
    triangulation.clear();
    Triangulation<dim> inner_triangulation;
    Triangulation<dim> inner_triangulation_2;
    GridGenerator::hyper_ball(inner_triangulation, Point<dim>(), inner_radius);
    inner_triangulation.refine_global(1);
    GridGenerator::flatten_triangulation(inner_triangulation,
                                         inner_triangulation_2);
    inner_triangulation_2.set_all_manifold_ids(0);
    inner_triangulation_2.set_all_manifold_ids_on_boundary(1);
    Triangulation<dim> outer_triangulation;

    GridGenerator::hyper_cube_with_cylindrical_hole(outer_triangulation,
                                                    inner_radius,
                                                    outer_radius);
    outer_triangulation.set_all_manifold_ids(0);
    outer_triangulation.set_all_manifold_ids_on_boundary(1);
    // Now merge the two triangulations
    GridGenerator::merge_triangulations(
      inner_triangulation_2, outer_triangulation, triangulation, 1e-3, true);
    triangulation.set_all_manifold_ids_on_boundary(0);
    triangulation.set_manifold(1, SphericalManifold<dim>(Point<dim>()));
    // triangulation.refine_global(2);
  });
  poisson.run();
}


TEST(Chapter2, Example2_2_Non_Matching_Interface)
{
  static const int        dim = 2;
  PDEs::MPI::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR
    "/book_prms/chapter_2_example_2_2_non_matching_balls.prm",
    "chapter_2_example_2_2_non_matching_balls.prm");

  poisson.run();
}