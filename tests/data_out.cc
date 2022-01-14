#include <deal.II/base/config.h>

#include "parsed_tools/data_out.h"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTester, DataOut)
{
  Triangulation<this->dim, this->spacedim> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<this->dim, this->spacedim>       fe(2);
  DoFHandler<this->dim, this->spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  Vector<double> solution(dof_handler.n_dofs());
  ParsedTools::DataOut<this->dim, this->spacedim> pdo(this->id());
  ParameterAcceptor::initialize();

  auto check_and_remove = [](const std::string &str) {
    ASSERT_TRUE(std::ifstream(str));
    std::remove(str.c_str());
  };

  auto test = [&](const std::string &fname, const std::string &extension) {
    this->parse("set Output format = " + extension, pdo);

    pdo.attach_dof_handler(dof_handler, "suffix");
    pdo.add_data_vector(solution, "solution");
    pdo.write_data_and_clear();
    check_and_remove(fname + "_suffix." + extension);
  };

  {
    std::stringstream ss;
    ss << "pdo_" << this->dim << "_" << this->spacedim;
    auto fname = ss.str();

    this->parse("set Problem base name = " + fname, pdo);

    test(fname, "vtk");
    test(fname, "vtu");
    // Vtu formats shouls also write a pvd file.
    check_and_remove(fname + ".pvd");
  }
}
