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

#include "compute_intersections.h"
#include "tests.h"
using namespace dealii;


template <int dim, int spacedim>
void
test()
{
  constexpr int                     degree = 4;
  constexpr double                  left   = -2;
  constexpr double                  right  = 0.;
  Triangulation<spacedim, spacedim> space_tria;
  Triangulation<dim, spacedim>      embedded_tria;

  GridGenerator::hyper_cube(space_tria, -1., 1.);
  GridGenerator::hyper_cube(embedded_tria, left, right);

  // Rotate the second grid.
  GridTools::rotate(M_PI_4, embedded_tria);



  DoFHandler<spacedim>      space_dh(space_tria);
  DoFHandler<dim, spacedim> embedded_dh(embedded_tria);

  FE_Q<spacedim>      fe_space(1);
  FE_Q<dim, spacedim> fe_embedded(1);

  space_dh.distribute_dofs(fe_space);
  embedded_dh.distribute_dofs(fe_embedded);

  {
    std::ofstream output_test_space("ambient.vtk");
    std::ofstream output_test_embedded("embedded.vtk");
    GridOut().write_vtk(space_tria, output_test_space);
    GridOut().write_vtk(embedded_tria, output_test_embedded);
  }
  const auto space_cache =
    std::make_unique<GridTools::Cache<spacedim>>(space_tria); // Q1 mapping
  const auto embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
    embedded_tria); // Q1 mapping

  // Compute Quadrature formulas on the intersections of the two grids, which
  // are just two squares, one rotated of M_PI degrees
  const auto vec_info =
    NonMatching::compute_intersection(*space_cache, *embedded_cache, degree);


  const auto &weights = std::get<2>(vec_info[0]).get_weights();



  deallog << "Expected: " << 1.0 << std::endl;
  deallog << "Obtained: "
          << std::accumulate(weights.cbegin(), weights.cend(), 0.) << std::endl;
}


int
main()
{
  initlog();


  test<2, 2>();
}
