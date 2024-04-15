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

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "pdes/babuska_bc.h"
#include "pdes/lagrange_multipliers.h"
#include "pdes/mixed_poisson.h"
#include "pdes/serial/poisson.h"
#include "pdes/stokes.h"

using namespace dealii;

TEST(Chapter1, Example1_1_P1_Regular)
{
  static const int           dim = 2;
  PDEs::Serial::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_1_1_p1_regular.prm",
    "chapter_1_example_1_1_p1_regular.prm");
  poisson.run();
}


TEST(Chapter1, Example1_1_P2_Regular)
{
  static const int           dim = 2;
  PDEs::Serial::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_1_2_p2_regular.prm",
    "chapter_1_example_1_2_p2_regular.prm");
  poisson.run();
}


TEST(Chapter1, Example1_3_Q2_Regular)
{
  static const int           dim = 2;
  PDEs::Serial::Poisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_1_3_q2_regular.prm",
    "chapter_1_example_1_3_q2_regular.prm");
  poisson.run();
}


TEST(Chapter1, Example1_4_Mixed_Poisson_MPI)
{
  static const int             dim = 2;
  PDEs::MPI::MixedPoisson<dim> poisson;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_1_4_mixed_poisson.prm",
    "chapter_1_example_1_4_mixed_poisson.prm");
  poisson.run();
}


TEST(Chapter1, Example1_5_Stokes_MPI)
{
  static const int       dim = 2;
  PDEs::MPI::Stokes<dim> stokes;
  ParameterAcceptor::initialize(FSI_SUITE_SOURCE_DIR
                                "/book_prms/chapter_1_example_1_5_stokes.prm",
                                "chapter_1_example_1_5_stokes.prm");
  stokes.run();
}


TEST(Chapter1, Example1_6_Stokes_Q2P1_MPI)
{
  static const int       dim = 2;
  PDEs::MPI::Stokes<dim> stokes;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_1_6_stokes_q2p1.prm",
    "chapter_1_example_1_6_stokes_q2p1.prm");
  stokes.run();
}



TEST(Chapter1, Example3_1_Babuska_BC_MPI)
{
  static const int                    dim = 2;
  PDEs::MPI::LagrangeMultipliers<dim> coupled;
  ParameterAcceptor::initialize(
    FSI_SUITE_SOURCE_DIR "/book_prms/chapter_1_example_3_1_babuska_bc.prm",
    "chapter_1_example_3_1_babuska_bc.prm");
  coupled.run();
}