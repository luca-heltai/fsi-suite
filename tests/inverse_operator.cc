#include "parsed_lac/inverse_operator.h"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace dealii;

TEST(InverseOperator, InvertPoisson)
{
  static const int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  MappingQ<dim>   mapping_q1(1);
  FE_Q<dim>       q1(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(q1);

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  dsp.compress();
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  sparsity_pattern.compress();

  SparseMatrix<double> A(sparsity_pattern);

  QGauss<dim> quadrature(2);
  MatrixCreator::create_mass_matrix(mapping_q1, dof_handler, quadrature, A);

  auto op_a = linear_operator<Vector<double>>(A);

  ParsedLAC::InverseOperator inverse_op("/", "cg");

  auto inv_a = inverse_op(op_a, PreconditionIdentity());

  Vector<double> u;
  op_a.reinit_domain_vector(u, true);
  for (unsigned int i = 0; i < u.size(); ++i)
    {
      u[i] = (double)(i + 1);
    }

  Vector<double> v     = inv_a * u;
  Vector<double> new_u = op_a * v;

  for (unsigned int i = 0; i < u.size(); ++i)
    {
      EXPECT_NEAR(u[i], new_u[i], 1e-12 * u.l2_norm());
    }
}