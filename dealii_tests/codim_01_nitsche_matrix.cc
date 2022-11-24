//-----------------------------------------------------------
//
//    Copyright (C) 2020 by the deal.II authors
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/sparse_matrix.h>

#include "assemble_nitsche_with_exact_intersections.h"
#include "compute_intersections.h"
#include "tests.h"
using namespace dealii;


template <int dim, int spacedim>
void
test()
{
  constexpr int                     degree = 4;
  constexpr double                  radius = .5;
  constexpr double                  left   = -.45;
  constexpr double                  right  = .55;
  Triangulation<spacedim, spacedim> space_tria;
  Triangulation<dim, spacedim>      embedded_tria;

  GridGenerator::hyper_cube(space_tria, -1., 1.);
  if constexpr (dim == 1 && spacedim == 2)
    {
      GridGenerator::hyper_sphere(embedded_tria, {.2, .2}, radius);
      space_tria.refine_global(2);
      embedded_tria.refine_global(6);
    }
  else if constexpr (dim == 2 && spacedim == 2)
    {
      GridGenerator::hyper_cube(embedded_tria, left, right);
      GridTools::rotate(M_PI_4 / 2., embedded_tria);
      space_tria.refine_global(4);
      embedded_tria.refine_global(2);
    }


  DoFHandler<spacedim>      space_dh(space_tria);
  DoFHandler<dim, spacedim> embedded_dh(embedded_tria);

  FE_Q<spacedim>      fe_space(1);
  FE_Q<dim, spacedim> fe_embedded(1);

  space_dh.distribute_dofs(fe_space);
  embedded_dh.distribute_dofs(fe_embedded);

  {
    std::string codim;
    (spacedim - dim == 0) ? codim = "codim_0" : codim = "codim_1";
    std::ofstream output_test_space("space_test_" + codim + ".vtk");
    std::ofstream output_test_embedded("embedded_test_" + codim + ".vtk");
    GridOut().write_vtk(space_tria, output_test_space);
    GridOut().write_vtk(embedded_tria, output_test_embedded);
  }
  auto space_cache =
    std::make_unique<GridTools::Cache<spacedim>>(space_tria); // Q1 mapping
  auto embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
    embedded_tria); // Q1 mapping

  // Compute Quadrature formulas on the intersections of the two
  const auto vec_info =
    NonMatching::compute_intersection(*space_cache, *embedded_cache, degree);


  AffineConstraints<double> space_constraints;


  SparsityPattern        sparsity_pattern;
  DynamicSparsityPattern dsp(space_dh.n_dofs());
  DoFTools::make_sparsity_pattern(space_dh, dsp);
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> nitsche_matrix(sparsity_pattern);



  const double h = space_dh.begin_active()->diameter();
  deallog << "h = " << h << '\n';
  NonMatching::
    assemble_nitsche_with_exact_intersections<spacedim, dim, spacedim>(
      space_dh,
      vec_info,
      nitsche_matrix,
      space_constraints,
      ComponentMask(),
      MappingQ1<spacedim>(),
      Functions::ConstantFunction<spacedim>(h));

  Vector<double> ones(space_dh.n_dofs());
  ones                = 1.0;
  const double result = nitsche_matrix.matrix_norm_square(ones);
  deallog << "Result with Nitsche matrix: " << std::setprecision(10) << result
          << std::endl;
  if constexpr (dim == 1 && spacedim == 2)
    {
      deallog << "Expected : " << std::setprecision(10) << 2. * M_PI * radius
              << std::endl;
    }
  else if constexpr (dim == 2 && spacedim == 2)
    {
      deallog << "Expected : " << std::setprecision(10)
              << std::pow(right - left, spacedim) << std::endl;
    }
}

int
main()
{
  initlog();


  test<1, 2>();
  test<2, 2>();
}
