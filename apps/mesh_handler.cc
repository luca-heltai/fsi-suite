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

template <int dim, int spacedim>
void
run(char **            argv,
    const std::string &input_parameter_file,
    const std::string &output_parameter_file)
{
  unsigned int verbosity = 2;
  deallog.depth_console(verbosity);
  ParameterAcceptor::prm.add_parameter("Verbosity", verbosity);
  ParsedTools::GridGenerator<dim, spacedim> pgg("/");
  // Exit if we were asked to print the help message
  if (Runner::setup_parameters_from_cli(argv,
                                        input_parameter_file,
                                        output_parameter_file) == -1)
    return;
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



int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console(1);

      const auto [dim, spacedim, input_parameter_file, output_parameter_file] =
        Runner::get_dimensions_and_parameter_files(argv);


      if (dim == 1 && spacedim == 1)
        run<1, 1>(argv, input_parameter_file, output_parameter_file);
      else if (dim == 1 && spacedim == 2)
        run<1, 2>(argv, input_parameter_file, output_parameter_file);
      else if (dim == 2 && spacedim == 2)
        run<2, 2>(argv, input_parameter_file, output_parameter_file);
      else if (dim == 2 && spacedim == 3)
        run<2, 3>(argv, input_parameter_file, output_parameter_file);
      else if (dim == 3 && spacedim == 3)
        run<3, 3>(argv, input_parameter_file, output_parameter_file);
      else
        {
          AssertThrow(false, ExcImpossibleInDimSpacedim(dim, spacedim));
        }
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
