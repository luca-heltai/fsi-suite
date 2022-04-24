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

// Build a polyhedron for each deal.II cell, and output it.

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <CGAL/IO/io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>

#include <fstream>

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Polyhedron_3<K>          Polyhedron;


#include "cgal/wrappers.h"
#include "tests.h"
template <int dim, int spacedim>
void
test()
{
  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};

  // unsigned int i = 0;
  for (const auto nv : d2t[dim])
    {
      Triangulation<dim, spacedim> tria;
      Polyhedron                   poly;

      const auto ref     = ReferenceCell::n_vertices_to_type(dim, nv);
      const auto mapping = ref.template get_default_mapping<dim, spacedim>(1);

      GridGenerator::reference_cell(tria, ref);

      const auto cell = tria.begin_active();
      CGALWrappers::to_cgal(cell, *mapping, poly);

      deallog << "dim: " << dim << ", spacedim: " << spacedim << std::endl;
      if (poly.is_valid())
        {
          deallog << "Valid polyhedron" << std::endl;
        }
      else
        {
          deallog << "Invalid polyhedron" << std::endl;
        }
      if (poly.is_closed())
        {
          deallog << "Closed polyhedron" << std::endl;
        }
      else
        {
          deallog << "Open polyhedron" << std::endl;
        }
      deallog << poly << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  initlog();
  test<1, 1>();
  test<1, 2>();
  test<1, 3>();
  test<2, 2>();
  test<2, 3>();
  test<3, 3>();
}
