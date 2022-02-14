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

#include "pdes/serial/reduced_lagrange.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
      std::string                      par_name = "reduced_lagrange_1d_2d.prm";
      if (argc > 1)
        par_name = argv[1];

      if (par_name.find("1d_2d") != std::string::npos)
        {
          PDEs::Serial::ReducedLagrange<1, 2> reduced_lagrange;
          ParameterAcceptor::initialize(
            par_name,
            "used_" + par_name,
            ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder);
          reduced_lagrange.run();
        }
      else if (par_name.find("2d_3d") != std::string::npos)
        {
          PDEs::Serial::ReducedLagrange<2, 3> reduced_lagrange;
          ParameterAcceptor::initialize(
            par_name,
            "used_" + par_name,
            ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder);
          reduced_lagrange.run();
        }
      else if (par_name.find("2d") != std::string::npos)
        {
          PDEs::Serial::ReducedLagrange<2, 2> reduced_lagrange;
          ParameterAcceptor::initialize(
            par_name,
            "used_" + par_name,
            ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder);
          reduced_lagrange.run();
        }
      else if (par_name.find("3d") != std::string::npos)
        {
          PDEs::Serial::ReducedLagrange<3, 3> reduced_lagrange;
          ParameterAcceptor::initialize(
            par_name,
            "used_" + par_name,
            ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder);
          reduced_lagrange.run();
        }
      else
        {
          AssertThrow(false,
                      ExcMessage(
                        "The parameter file name should contain either "
                        "1d_2d, 2d_3d, 2d, or 3d"));
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
