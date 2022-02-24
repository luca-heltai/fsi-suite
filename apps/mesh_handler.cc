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

/**
 * Mesh generator and reader.
 *
 * @ingroup basics
 * @file mesh_handler.cc
 *
 * This program is useful to debug input grid files, to convert from one format
 * to another, or simply to generate and view one of the internal deal.II grids.
 *
 * The mesh_handler executable can be driven by a configuration file, or by
 * command line arguments.
 */

#include <deal.II/base/utilities.h>

#include <deal.II/grid/reference_cell.h>

#include "argh.hpp"
#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_info.h"
#include "runner.h"

using namespace dealii;

template <int dim, int spacedim = dim>
class MeshHandler : public ParameterAcceptor
{
public:
  MeshHandler()
    : ParameterAcceptor("/")
    , pgg("/")
  {
    add_parameter("Verbosity", verbosity);
  }

  void
  run()
  {
    deallog.depth_console(verbosity);
    Triangulation<dim, spacedim> tria;
    pgg.generate(tria);
    pgg.write(tria);
    ParsedTools::GridInfo info(tria, verbosity);
    deallog << "=================" << std::endl;
    deallog << "Used parameters: " << std::endl;
    deallog << "=================" << std::endl;
    ParameterAcceptor::prm.log_parameters(deallog);
    deallog << "=================" << std::endl;
    deallog << "Grid information: " << std::endl;
    deallog << "=================" << std::endl;
    info.print_info(deallog);
  }

private:
  unsigned int                              verbosity = 2;
  ParsedTools::GridGenerator<dim, spacedim> pgg;
};



int
main(int argc, char **argv)
{
  RUNNER(MeshHandler, argc, argv);
}