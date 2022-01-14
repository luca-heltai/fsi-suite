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