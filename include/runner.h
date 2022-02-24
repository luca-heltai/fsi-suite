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

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include "argh.hpp"
#include "text_flow.hpp"

using namespace dealii;

/**
 * Parse from the command line the parameter file names (both input and output)
 * and the running dimensions (both dim and spacedim).
 *
 * This function uses the command line interface parsers from the argh library
 * (https://github.com/adishavit/argh), and recognizes the following parameters
 * from the command line:
 *
 * @code {.sh}
 * ./your_code_here --help
 * Usage: ./your_code_here [OPTIONS] [PRM FILE]
 * Options:
 * -h, --help                        Print this help message
 * -p, --prm_file <filename>         Input parameter file. Defaults to the
 *                                   empty string (meaning: use the default
 *                                   values, or the values specified on the
 *                                   command line). Notice that you can
 *                                   specify the input paramter file also as
 *                                   the first positional arguement to the
 *                                   program.
 * -o, --output_prm_file <filename>  Where to write the file containing the
 *                                   actual parameters used in this run of
 *                                   the program. It defaults to the string
 *                                   `used_./mesh_handler' followed by a
 *                                   string of the type '1d_2d' containing
 *                                   the dimension and the spacedimension at
 *                                   which the program was run if the input
 *                                   parameter file is not specified,
 *                                   otherwise it defaults to the string
 *                                   `used_' followed by the name of the
 *                                   input parameter file.
 * -d, --dim <value>                 Dimension at which this program should
 *                                   be run. Defaults to 2.
 * -s, --spacedim <value>            Space dimension at which this program
 *                                   should run. Defaults to 2.
 * -"Section/option name"=<value>    Any of the options that you can specify
 *                                   in the parameter file. The format here
 *                                   is the following: -"Section/Subsection/
 *                                   option"="value,another value", where the
 *                                   quotes are required only if an otpion
 *                                   contains spaces, or if a value contains
 *                                   separators, like commas, columns, etc.
 * @endcode
 *
 * @warning This function is usually called before any class initialization
 * takes place. You should use this function in conjunction with its companion
 * function setup_parameters_from_cli() *after* you have initialized your
 * classes and ParameterAcceptor::initialize() has been called at least once.
 *
 * Example usage (assuming you have a class called MyClass derived from
 * ParameterAcceptor):
 * @code
 * int main(int argc, char **argv) {
 *  auto [dim, spacedim, infile, outfile] =
 *       get_dimensions_and_parameter_files(argv);
 *  // do something with dim and spacedim...
 *  MyClass<dimpar, spacedimpar> my_class;
 *  ParameterAcceptor::initialize();
 *  // Check we were not asked to print help message
 *  if(setup_parameters_from_cli(argv, infile, outfile) == -1)
 *    return 0;
 *  my_class.run();
 * }
 * @endcode
 *
 * @param argv Arguments of the command line.
 *
 * @return (dim, spacedim, input_parameters, output_parameters) Desired
 * dimension and spacedimesion of the problem to run, and input and output
 * parameter filenames.
 */
std::tuple<int, int, std::string, std::string>
get_dimensions_and_parameter_files(char **argv)
{
  argh::parser cli(
    {"p", "prm_file", "d", "dim", "s", "spacedim", "o", "output_prm_file"});
  cli.parse(argv);
  int         dim                   = 2;
  int         spacedim              = 2;
  std::string input_parameter_file  = "";
  std::string output_parameter_file = "";

  std::string exename = cli(0).str().substr(cli(0).str().find_last_of("/") + 1);


  // Either as -p prm_file or as first positional argument after all options
  cli({"p", "prm_file"}, input_parameter_file) >> input_parameter_file;
  cli(1, input_parameter_file) >> input_parameter_file;

  cli({"d", "dim"}, 2) >> dim;
  cli({"s", "spacedim"}, 2) >> spacedim;

  if (input_parameter_file != "")
    {
      output_parameter_file = "used_" + input_parameter_file;
    }
  else if (dim == spacedim)
    {
      output_parameter_file =
        "used_" + exename + "_" + std::to_string(dim) + "d.prm";
    }
  else
    {
      output_parameter_file = "used_" + exename + "_" + std::to_string(dim) +
                              "d_" + std::to_string(spacedim) + "d.prm";
    }

  // If you want to overwrite the output parameter file, use the -o option
  cli({"o", "output_prm_file"}, output_parameter_file) >> output_parameter_file;

  deallog << "Will run in dimension " << dim << " and spacedimemsion "
          << spacedim << std::endl
          << "Input parameter file: " << input_parameter_file << std::endl
          << "Output parameter file: " << output_parameter_file << std::endl;

  return std::make_tuple(dim,
                         spacedim,
                         input_parameter_file,
                         output_parameter_file);
}


/**
 * Setup the ParameterAcceptor::prm according to the parameters specified in the
 * parameter file, and the parameters specified from the command line.
 *
 * This function uses the command line interface parsers from the argh library
 * (https://github.com/adishavit/argh), and allows you to specify input
 * parameter files, output parameter files, and to change any option recognized
 * by the parameter file itself from the command line.
 *
 * This function is usually used in conjunction with the function
 * get_dimensions_and_parameter_files(), in order to ask from the command line
 * what simulation to run (1d, 2d, 3d, etc.), and what input parameter files to
 * read from. These are then passed to this function, **after** you have called
 * at least once ParameterAcceptor::initialize() on your classes (which you will
 * have instantiated with the correct dimension thanks to the function above).
 *
 * Example usage (assuming you have a class called MyClass derived from
 * ParameterAcceptor):
 * @code
 * int main(int argc, char **argv) {
 *  auto [dim, spacedim, infile, outfile] =
 *       get_dimensions_and_parameter_files(argv);
 *  // do something with dim and spacedim...
 *  if(dim == 2 && spacedim == 2) {
 *    MyClass<2, 2> my_class;
 *    ParameterAcceptor::initialize();
 *    // Check we were not asked to print help message
 *    if(setup_parameters_from_cli(argv, infile, outfile) == -1)
 *      return 0;
 *    my_class.run();
 *  }
 * }
 * @endcode
 *
 * If the option `-h` or `--help` is found on the command line, this function
 * outputs the following help message:
 * @code {.sh}
 * ./your_code_here --help
 * Usage: ./your_code_here [OPTIONS] [PRM FILE]
 * Options:
 * -h, --help                        Print this help message
 * -p, --prm_file <filename>         Input parameter file. Defaults to the
 *                                   empty string (meaning: use the default
 *                                   values, or the values specified on the
 *                                   command line). Notice that you can
 *                                   specify the input paramter file also as
 *                                   the first positional arguement to the
 *                                   program.
 * -o, --output_prm_file <filename>  Where to write the file containing the
 *                                   actual parameters used in this run of
 *                                   the program. It defaults to the string
 *                                   `used_./mesh_handler' followed by a
 *                                   string of the type '1d_2d' containing
 *                                   the dimension and the spacedimension at
 *                                   which the program was run if the input
 *                                   parameter file is not specified,
 *                                   otherwise it defaults to the string
 *                                   `used_' followed by the name of the
 *                                   input parameter file.
 * -d, --dim <value>                 Dimension at which this program should
 *                                   be run. Defaults to 2.
 * -s, --spacedim <value>            Space dimension at which this program
 *                                   should run. Defaults to 2.
 * -"Section/option name"=<value>    Any of the options that you can specify
 *                                   in the parameter file. The format here
 *                                   is the following: -"Section/Subsection/
 *                                   option"="value,another value", where the
 *                                   quotes are required only if an otpion
 *                                   contains spaces, or if a value contains
 *                                   separators, like commas, columns, etc.
 * @endcode
 * followed by a list of the options that are recognized by the parameter file,
 * i.e., for the mesh_handler.cc application, we get:
 * @code {.sh}
 * ...
 * -"Section/option name"=<value>     Any of the options that you can specify
 *                                    in the parameter file. The format here
 *                                    is the following: -"Section/Subsection/
 *                                    option"="value,another value", where the
 *                                    quotes are required only if an otpion
 *                                    contains spaces, or if a value contains
 *                                    separators, like commas, columns, etc.
 *
 * Listing of Parameters:
 *
 * set Arguments                  =   Any string
 * set Initial grid refinement    =   An integer
 * set Input name                 =   Any string
 * set Output name                =   Any string
 * set Transform to simplex grid  =   A boolean value (true or false)
 * set Verbosity                  =   An integer n such that -1 <= n <=
 * 2147483647
 * @endcode
 *
 * @return 0: everything is fine, 1: unused parameters, -1: help printed
 */
int
setup_parameters_from_cli(char **            argv,
                          const std::string &input_parameter_file,
                          const std::string &output_parameter_file)
{
  argh::parser cli(argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
  ParameterAcceptor::initialize(input_parameter_file, output_parameter_file);

  if (cli[{"h", "help"}])
    {
      auto format = [](const auto &a, const auto &b) {
        return TextFlow::Column(a).width(30) + TextFlow::Column(b).width(50);
      };

      std::cout
        << "Usage: " << cli(0).str() << " [OPTIONS] [PRM FILE]" << std::endl
        << "Options:" << std::endl
        << format("-h, --help", "Print this help message") << std::endl
        << format(
             "-p, --prm_file <filename>",
             "Input parameter file. Defaults"
             " to the empty string (meaning: use the default values, or the "
             "values specified on the command line). Notice that you can "
             "specify the input paramter file also as the first positional arguement to the program.")
        << std::endl
        << format(
             "-o, --output_prm_file <filename>",
             "Where to write the file containing the actual parameters "
             "used in this run of the program. It defaults to the string `used_" +
               cli(0).str() +
               "' followed by a string of the type '1d_2d' "
               "containing the dimension and the spacedimension at which the "
               "program was run if the input parameter file is not specified, "
               "otherwise it defaults to the string `used_' followed by the "
               "name of the input parameter file.")
        << std::endl
        << format(
             "-d, --dim <value>",
             "Dimension at which this program should be run. Defaults to 2.")
        << std::endl
        << format(
             "-s, --spacedim <value>",
             "Space dimension at which this program should run. Defaults to 2.")
        << std::endl
        << format(
             "-\"Section/option name\"=<value>",
             "Any of the options that you can specify in the parameter file. "
             "The format here is the following: "
             "-\"Section/Subsection/option\"=\"value,another value\", "
             "where the quotes are required only if an otpion contains spaces, "
             "or if a value contains separators, like commas, columns, etc.")
        << std::endl
        << std::endl;
      ParameterAcceptor::prm.print_parameters(std::cout,
                                              ParameterHandler::Description);
      return -1;
    }

  std::set<std::string> non_prm{"h",
                                "help",
                                "p",
                                "prm_file",
                                "o",
                                "output_prm_file",
                                "d",
                                "dim",
                                "s",
                                "spacedim"};
  for (auto &p : cli.params())
    if (non_prm.find(p.first) == non_prm.end())
      {
        auto              path  = Utilities::split_string_list(p.first, "/");
        const std::string entry = path.back();
        path.pop_back();

        for (const auto &sec : path)
          ParameterAcceptor::prm.enter_subsection(sec);

        ParameterAcceptor::prm.set(entry, p.second);

        for (const auto &sec : path)
          {
            (void)sec;
            ParameterAcceptor::prm.leave_subsection();
          }
      }
  int ret = 0;
  for (auto &p : cli.pos_args())
    if (p != argv[0] && p != input_parameter_file)
      {
        deallog << "WARNING -- ignoring positional argument: " << p
                << std::endl;
        ret = 1;
      }

  ParameterAcceptor::initialize("",
                                output_parameter_file,
                                ParameterHandler::Short |
                                  ParameterHandler::KeepDeclarationOrder);
  // Everything went fine, so return 0 or 1
  return ret;
}



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