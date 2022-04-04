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

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria_accessor.h>

#include <memory>

#include "compute_intersections.h"
#include "tests.h"
using namespace dealii;


template <int dim, int spacedim>
void
test()
{
  constexpr int                     degree = 3;
  Triangulation<spacedim, spacedim> space_tria;
  Triangulation<dim, spacedim>      embedded_tria;

  GridGenerator::hyper_cube(space_tria, -1., 1.);
  GridGenerator::hyper_cube(embedded_tria, -0.45, 0.25);
  space_tria.refine_global(2);
  embedded_tria.refine_global(2);


  DoFHandler<spacedim>      space_dh(space_tria);
  DoFHandler<dim, spacedim> embedded_dh(embedded_tria);

  FE_Q<spacedim>      fe_space(1);
  FE_Q<dim, spacedim> fe_embedded(1);

  space_dh.distribute_dofs(fe_space);
  embedded_dh.distribute_dofs(fe_embedded);


  auto space_cache =
    std::make_unique<GridTools::Cache<spacedim>>(space_tria); // Q1 mapping
  auto embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
    embedded_tria); // Q1 mapping

  // Compute Quadrature formulas on the intersections of the two
  const auto vec_info =
    NonMatching::compute_intersection(*space_cache, *embedded_cache, degree);

  assert(vec_info.size() == 25); // Exact number of intersections


  std::ofstream output_test_space("space_test.vtk");
  std::ofstream output_test_embedded("embedded_test.vtk");
  GridOut().write_vtk(space_tria, output_test_space);
  GridOut().write_vtk(embedded_tria, output_test_embedded);
  // Print cells ids and points are the printed


  for (const auto &infos : vec_info)
    {
      const auto &[cell0, cell1, quad_form] = infos;
      deallog << "Idx Cell0: " << cell0->active_cell_index() << std::endl;
      deallog << "Idx Cell1: " << cell1->active_cell_index() << std::endl;
      deallog << "Quad pts" << std::endl;
      for (const auto &q_pts : quad_form.get_points())
        {
          deallog << q_pts << std::endl;
        }
    }
}

int
main()
{
  initlog();
  test<2, 2>();
}
