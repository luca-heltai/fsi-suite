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

#include "runner.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include "argh.hpp"
#include "text_flow.hpp"

using namespace dealii;

namespace Runner
{
  /**
   * \brief Retrieves the dimension and space dimension from a parameter file.
   *
   * This function reads a parameter file and retrieves the values of the
   * "dimension" and "space dimension" parameters. If the reading of the input
   * file fails, it returns the default values provided.
   *
   * @p prm_file The path to the parameter file. @p default_dim The
   * default value for the dimension. @p default_spacedim The default value
   * for the space dimension. \return A pair containing the dimension and space
   * dimension.
   *
   * \throws std::exception If an error occurs while parsing the input file.
   */
  std::pair<unsigned int, unsigned int>
  get_dimension_and_spacedimension(const std::string &prm_file,
                                   const unsigned int default_dim      = 2,
                                   const unsigned int default_spacedim = 2)
  {
    ParameterAcceptor::prm.declare_entry("dim",
                                         std::to_string(default_dim),
                                         Patterns::Integer(1, 3));
    ParameterAcceptor::prm.declare_entry("spacedim",
                                         std::to_string(default_spacedim),
                                         Patterns::Integer(1, 3));
    try
      {
        ParameterAcceptor::prm.parse_input(prm_file, "", true);
        auto dim      = ParameterAcceptor::prm.get_integer("dim");
        auto spacedim = ParameterAcceptor::prm.get_integer("spacedim");
        return {dim, spacedim};
      }
    catch (std::exception &exc)
      {
        return {default_dim, default_spacedim};
        throw;
      }
  }


  /**
   * Retrieves the dimensions and parameter files from the command line
   * arguments.
   *
   * This function parses the command line arguments and extracts the dimensions
   * and parameter file names. It supports both options and positional arguments
   * for specifying the input parameter file. It also checks if the dimensions
   * specified in the parameter file and the command line arguments match.
   *
   * @param argv The command line arguments.
   * @return A tuple containing the dimensions and parameter file names.
   *         The tuple elements are in the following order:
   *         - int: The dimension.
   *         - int: The space dimension.
   *         - std::string: The input parameter file name.
   *         - std::string: The output parameter file name.
   */
  std::tuple<int, int, std::string, std::string>
  get_dimensions_and_parameter_files(char **argv)
  {
    argh::parser cli(argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    unsigned int dim                   = 2;
    unsigned int spacedim              = 2;
    std::string  input_parameter_file  = "";
    std::string  output_parameter_file = "";

    const std::string exename =
      cli(0).str().substr(cli(0).str().find_last_of("/") + 1);


    // Either as -i prm_file or as first positional argument after all options
    cli({"i", "input_prm_file"}, input_parameter_file) >> input_parameter_file;
    cli(1, input_parameter_file) >> input_parameter_file;

    const bool has_dim_or_spacedim =
      cli[{"d", "dim"}] || cli[{"s", "spacedim"}];
    // Now read from command line the dimension and space dimension
    cli({"d", "dim"}) >> dim;
    // Make sure the default is to set spacedim = dim
    cli({"s", "spacedim"}, dim) >> spacedim;

    // And do the same from the parameter file
    const auto [prm_dim, prm_spacedim] =
      get_dimension_and_spacedimension(input_parameter_file, dim, spacedim);

    // Throw an exception if the inputer parameter file and the command line do
    // not agree. Notice that, if the file doees not exist, they will agree,
    // since the default values are the same.
    AssertThrow(
      !has_dim_or_spacedim || (dim == prm_dim) && (spacedim == prm_spacedim),
      dealii::ExcMessage(
        "You have specified a parameter file that contains a specification "
        "of the dimension and of the space dimension, as <" +
        std::to_string(prm_dim) + ", " + std::to_string(prm_spacedim) +
        ">, but you also indicated a -d (--dim) = " + std::to_string(dim) +
        " or -s (--spacedim) = " + std::to_string(spacedim) +
        " argument on the command line that do not match the content of the parameter file. "
        "Use only one of the two ways to select the dimension and the "
        "space dimension, or make sure that what you specify in the parameter file "
        "matches what you specify on the command line."));

    // Now the logic to deduce the output parameter file name. Make sure we
    // output in the current directory, even if the file is specified with a
    // full path
    if (input_parameter_file != "")
      {
        auto rel_name = input_parameter_file.substr(
          input_parameter_file.find_last_of("/") + 1);
        output_parameter_file = "used_" + rel_name;
      }
    else
      {
        output_parameter_file = "used_" + exename + ".prm";
      }

    // If you want to overwrite the output parameter file, use the -o option
    cli({"o", "output_prm_file"}, output_parameter_file) >>
      output_parameter_file;

    deallog << "Will run in dimension " << dim << " and spacedimension "
            << spacedim << std::endl
            << "Input parameter file: " << input_parameter_file << std::endl
            << "Output parameter file: " << output_parameter_file << std::endl;

    return std::make_tuple(prm_dim,
                           prm_spacedim,
                           input_parameter_file,
                           output_parameter_file);
  }


  /**
   * @brief Sets up the program parameters from the command-line arguments.
   *
   * This function parses the command-line arguments using the `argh` library
   * and sets up the program parameters accordingly. It initializes the
   * `ParameterAcceptor` with the input and output parameter file paths. If the
   * `-h` or `--help` option is provided, it prints the help message and the
   * list of available options. The function also handles setting the values for
   * options specified in the command-line arguments. It ignores any positional
   * arguments other than the program name and the input parameter file. After
   * setting up the parameters, it initializes the `ParameterAcceptor` again
   * with the output parameter file path. If the `--pause` option is provided
   * and the current MPI process is the root process, it waits for a keypress
   * before continuing.
   *
   * @param argv The command-line arguments.
   * @param input_parameter_file The path to the input parameter file.
   * @param output_parameter_file The path to the output parameter file.
   * @return Returns 0 if everything went fine, or 1 if there were any warnings or errors.
   */
  int
  setup_parameters_from_cli(char             **argv,
                            const std::string &input_parameter_file,
                            const std::string &output_parameter_file)
  {
    argh::parser cli(argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    ParameterAcceptor::initialize(input_parameter_file, output_parameter_file);

    if (cli[{"h", "help"}])
      {
        auto format = [](const auto &a, const auto &b) {
          return TextFlow::Column(a).width(34) + TextFlow::Column(b).width(46);
        };


        const std::string exename =
          cli(0).str().substr(cli(0).str().find_last_of("/") + 1);

        std::cout
          << "Usage: " << cli(0).str() << " [OPTIONS] [PRM FILE]" << std::endl
          << "Options:" << std::endl
          << format("-h, --help", "Print this help message") << std::endl
          << format(
               "-i, --input_prm_file <filename>",
               "Input parameter file. Defaults"
               " to the empty string (meaning: use the default values, or the "
               "values specified on the command line). Notice that you can "
               "specify the input paramter file also as the first positional arguement to the program.")
          << std::endl
          << format(
               "-o, --output_prm_file <filename>",
               "Where to write the file containing the actual parameters "
               "used in this run of the program. It defaults to the string `used_" +
                 exename +
                 "' if the input parameter file is not specified, "
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
          << format("-pause", "Wait for a keypress to attach a debugger.")
          << std::endl
          << std::endl;
        ParameterAcceptor::prm.print_parameters(std::cout,
                                                ParameterHandler::Description);
        return -1;
      }

    std::set<std::string> non_prm{"h",
                                  "help",
                                  "i",
                                  "input_prm_file",
                                  "o",
                                  "output_prm_file",
                                  "d",
                                  "dim",
                                  "s",
                                  "spacedim",
                                  "pause"};
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
    // Check if we need to wait for input
    if (cli["pause"] && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "============================================" << std::endl
                  << "PID: " << getpid() << std::endl
                  << "============================================" << std::endl
                  << "Press any key to continue..." << std::endl
                  << "============================================"
                  << std::endl;
        std::cin.get();
      }

    // Everything went fine, so return 0 or 1
    return ret;
  }
} // namespace Runner