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

#ifndef fsi_runner_h
#define fsi_runner_h

#include <deal.II/base/utilities.h>

using namespace dealii;

#ifndef DOXYGEN

#  define RUN_DIM(ClassName, par_name)                                       \
    if (par_name.find("1d") != std::string::npos)                            \
      {                                                                      \
        ClassName<1> class_name;                                             \
        ParameterAcceptor::initialize(                                       \
          par_name,                                                          \
          "used_" + par_name,                                                \
          ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder); \
        class_name.run();                                                    \
      }                                                                      \
    else                                                                     \
      RUN_DIM_NO_ONE(ClassName, par_name)

#  define RUN_DIM_NO_ONE(ClassName, par_name)                                \
    if (par_name.find("2d") != std::string::npos)                            \
      {                                                                      \
        ClassName<2> class_name;                                             \
        ParameterAcceptor::initialize(                                       \
          par_name,                                                          \
          "used_" + par_name,                                                \
          ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder); \
        class_name.run();                                                    \
      }                                                                      \
    else if (par_name.find("3d") != std::string::npos)                       \
      {                                                                      \
        ClassName<3> class_name;                                             \
        ParameterAcceptor::initialize(                                       \
          par_name,                                                          \
          "used_" + par_name,                                                \
          ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder); \
        class_name.run();                                                    \
      }

// Run all co-dimension one cases
#  define RUN_DIM_SPACEDIM(ClassName, par_name)                              \
    if (par_name.find("1d_2d") != std::string::npos)                         \
      {                                                                      \
        ClassName<1, 2> class_name;                                          \
        ParameterAcceptor::initialize(                                       \
          par_name,                                                          \
          "used_" + par_name,                                                \
          ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder); \
        class_name.run();                                                    \
      }                                                                      \
    else if (par_name.find("2d_3d") != std::string::npos)                    \
      {                                                                      \
        ClassName<2, 3> class_name;                                          \
        ParameterAcceptor::initialize(                                       \
          par_name,                                                          \
          "used_" + par_name,                                                \
          ParameterHandler::Short | ParameterHandler::KeepDeclarationOrder); \
        class_name.run();                                                    \
      }

// Standard catch block for all snippets
#  define STANDARD_CATCH()                                                \
    catch (std::exception & exc)                                          \
    {                                                                     \
      std::cerr << std::endl                                              \
                << std::endl                                              \
                << "----------------------------------------------------" \
                << std::endl;                                             \
      std::cerr << "Exception on processing: " << std::endl               \
                << exc.what() << std::endl                                \
                << "Aborting!" << std::endl                               \
                << "----------------------------------------------------" \
                << std::endl;                                             \
      return 1;                                                           \
    }                                                                     \
    catch (...)                                                           \
    {                                                                     \
      std::cerr << std::endl                                              \
                << std::endl                                              \
                << "----------------------------------------------------" \
                << std::endl;                                             \
      std::cerr << "Unknown exception!" << std::endl                      \
                << "Aborting!" << std::endl                               \
                << "----------------------------------------------------" \
                << std::endl;                                             \
      return 1;                                                           \
    }

#endif


/**
 * Body of the main function for a program that runs a simulation in dimension
 * dim only using the ParameterAcceptor class, dimensions 1, 2, and 3.
 */
#define RUNNER_DIM(ClassName, ParameterName, argc, argv)                   \
  try                                                                      \
    {                                                                      \
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);     \
      std::string                      par_name = ParameterName "_2d.prm"; \
      if (argc > 1)                                                        \
        par_name = argv[1];                                                \
                                                                           \
      RUN_DIM(ClassName, par_name) else                                    \
      {                                                                    \
        AssertThrow(false,                                                 \
                    ExcMessage("The parameter file name should contain "   \
                               "either 1d, 2d, or 3d"));                   \
      }                                                                    \
    }                                                                      \
  STANDARD_CATCH()

/**
 * Body of the main function for a program that runs a simulation in dimensions
 * two or three using the ParameterAcceptor class. Not instantiated for
 * dimension one.
 */
#define RUNNER_DIM_NO_ONE(ClassName, ParameterName, argc, argv)            \
  try                                                                      \
    {                                                                      \
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);     \
      std::string                      par_name = ParameterName "_2d.prm"; \
      if (argc > 1)                                                        \
        par_name = argv[1];                                                \
                                                                           \
      RUN_DIM_NO_ONE(ClassName, par_name) else                             \
      {                                                                    \
        AssertThrow(false,                                                 \
                    ExcMessage("The parameter file name should contain "   \
                               "either 2d or 3d"));                        \
      }                                                                    \
    }                                                                      \
  STANDARD_CATCH()

/**
 * Body of the main function for a program that runs a simulation in dim and
 * spacedim, using the ParameterAcceptor class. Instantiate for <1,1>, <1,2>,
 * <2,2>, <2,3>, and <3,3>
 */
#define RUNNER_DIM_SPACEDIM(ClassName, ParameterName, argc, argv)          \
  try                                                                      \
    {                                                                      \
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);     \
      std::string                      par_name = ParameterName "_2d.prm"; \
      if (argc > 1)                                                        \
        par_name = argv[1];                                                \
                                                                           \
      RUN_DIM_SPACEDIM(ClassName, par_name)                                \
      else RUN_DIM(ClassName, par_name) else                               \
      {                                                                    \
        AssertThrow(false,                                                 \
                    ExcMessage("The parameter file name should contain "   \
                               "either  1d, 1d_2d, 2d, 2d_3d, or 3d"));    \
      }                                                                    \
    }                                                                      \
  STANDARD_CATCH()

/**
 * Body of the main function for a program that runs a simulation in dim and
 * spacedim, using the ParameterAcceptor class. Instantiate for <1,2>, <2,2>,
 * <2,3>, and <3,3>
 */
#define RUNNER_DIM_SPACEDIM_NO_ONE(ClassName, ParameterName, argc, argv)   \
  try                                                                      \
    {                                                                      \
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);     \
      std::string                      par_name = ParameterName "_2d.prm"; \
      if (argc > 1)                                                        \
        par_name = argv[1];                                                \
                                                                           \
      RUN_DIM_SPACEDIM(ClassName, par_name)                                \
      else RUN_DIM_NO_ONE(ClassName, par_name) else                        \
      {                                                                    \
        AssertThrow(false,                                                 \
                    ExcMessage("The parameter file name should contain "   \
                               "either 1d_2d, 2d_3d, 2d, or 3d"));         \
      }                                                                    \
    }                                                                      \
  STANDARD_CATCH()

#endif