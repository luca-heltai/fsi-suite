#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"
#include "tools/parsed_preconditioner/amg.h"
#include "tools/parsed_preconditioner/amg_muelu.h"
#include "tools/parsed_preconditioner/ilu.h"
#include "tools/parsed_preconditioner/jacobi.h"

using namespace dealii;

TEST(Preconditioners, Instantiate)
{
  Tools::ParsedAMGPreconditioner      amg("AMG");
  Tools::ParsedAMGMueLuPreconditioner amg_muelu("AMG Muelu");
  Tools::ParsedILUPreconditioner      ilu("ILU");
  Tools::ParsedJacobiPreconditioner   jacobi("Jacobi");

  ParameterAcceptor::initialize();
}