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
 * This program is useful to debug input grid files. It gathers information from
 * the file specified in the input parameter and prints it on screen. It is
 * based on step-1 of the deal.II library, and offers a general overview of the
 * class ParsedTools::GridGenerator.
 */

#include <deal.II/grid/reference_cell.h>

#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_info.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      par_name   = "";
      unsigned int                     info_level = 0;
      if (argc > 1)
        par_name = argv[1];
      if (argc > 2)
        info_level = std::atoi(argv[2]);


      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      ParsedTools::GridGenerator<2> pgg;
      ParameterAcceptor::initialize(par_name);
      Triangulation<2> tria;
      pgg.generate(tria);
      pgg.write(tria);
      ParsedTools::GridInfo info(tria, info_level);
      info.print_info(deallog);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
