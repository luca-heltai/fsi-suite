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

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "parsed_lac/amg.h"
#include "parsed_lac/amg_muelu.h"
#include "parsed_lac/ilu.h"
#include "parsed_lac/jacobi.h"

using namespace dealii;

TEST(Preconditioners, Instantiate)
{
  ParsedLAC::AMGPreconditioner      amg("AMG");
  ParsedLAC::AMGMueLuPreconditioner amg_muelu("AMG Muelu");
  ParsedLAC::ILUPreconditioner      ilu("ILU");
  ParsedLAC::JacobiPreconditioner   jacobi("Jacobi");

  ParameterAcceptor::initialize();
}