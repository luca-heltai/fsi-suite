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

#include "parsed_tools/non_matching_coupling.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;


TYPED_TEST(DimSpacedimTester, SerialNonMatchingCoupling)
{
  constexpr int dim0     = TestFixture::spacedim;
  constexpr int dim1     = TestFixture::dim;
  constexpr int spacedim = TestFixture::spacedim;

  Triangulation<dim0, spacedim> tria0;
  Triangulation<dim1, spacedim> tria1;

  GridGenerator::hyper_cube(tria0, -1, 1);
  GridGenerator::hyper_cube(tria1, -.44444, .3333);

  tria0.refine_global((dim0 < 3 ? 3 : 2));
  tria1.refine_global(2);

  for (unsigned int i = 0; i < 2; ++i)
    {
      for (const auto &cell : tria1.active_cell_iterators())
        if (cell->center()[0] < 0)
          cell->set_refine_flag();
      tria1.execute_coarsening_and_refinement();

      for (const auto &cell : tria0.active_cell_iterators())
        if (cell->center()[0] > 0)
          cell->set_refine_flag();
      tria0.execute_coarsening_and_refinement();
    }

  FE_Q<dim0, spacedim> fe0(1);
  FE_Q<dim1, spacedim> fe1(1);

  DoFHandler<dim0, spacedim> dh0(tria0);
  DoFHandler<dim1, spacedim> dh1(tria1);

  GridTools::Cache<dim0, spacedim> cache0(tria0);
  GridTools::Cache<dim1, spacedim> cache1(tria1);

  dh0.distribute_dofs(fe0);
  dh1.distribute_dofs(fe1);

  AffineConstraints<double> constraints0;
  AffineConstraints<double> constraints1;

  DoFTools::make_hanging_node_constraints(dh0, constraints0);
  DoFTools::make_zero_boundary_constraints(dh0, constraints0);

  DoFTools::make_hanging_node_constraints(dh1, constraints1);

  constraints0.close();
  constraints1.close();

  ParsedTools::NonMatchingCoupling<dim1, dim0> coupling(this->id());
  coupling.initialize(cache0, dh0, constraints0, cache1, dh1, constraints1);

  SparsityPattern sparsity;
  coupling.assemble_sparsity(sparsity);
  SparseMatrix<double> coupling_matrix(sparsity);
  coupling.assemble_matrix(coupling_matrix);

  SparsityPattern mass_sparsity1;
  {
    DynamicSparsityPattern dsp(dh1.n_dofs(), dh1.n_dofs());
    DoFTools::make_sparsity_pattern(dh1, dsp, constraints1, false);
    mass_sparsity1.copy_from(dsp);
  }
  SparseMatrix<double> mass_matrix1(mass_sparsity1);
  MatrixTools::create_mass_matrix(dh1,
                                  QGauss<dim1>(2),
                                  mass_matrix1,
                                  static_cast<const Function<spacedim> *>(
                                    nullptr),
                                  constraints1);

  SparseDirectUMFPACK mass_matrix1_inv;
  mass_matrix1_inv.factorize(mass_matrix1);

  // now take ones in dh0, project them onto dh1,
  // get back ones, and check for the error.
  //
  // WARNINGS: Only works if dh1 is immersed in dh0

  Vector<double> ones0(dh0.n_dofs());
  Vector<double> ones1(dh1.n_dofs());

  ones0 = 1.0;
  constraints0.distribute(ones0);

  coupling_matrix.Tvmult(ones1, ones0);
  mass_matrix1_inv.solve(ones1);
  constraints1.distribute(ones1);

  Vector<double> exact_ones1(dh1.n_dofs());
  exact_ones1 = 1.0;
  constraints1.distribute(exact_ones1);
  ones1 -= exact_ones1;

  ASSERT_NEAR(ones1.linfty_norm(), 0.0, 1e-10);
}
