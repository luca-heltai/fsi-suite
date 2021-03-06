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
  std::tuple<int, int, std::string, std::string>
  get_dimensions_and_parameter_files(char **argv)
  {
    argh::parser cli(argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    int          dim                   = 2;
    int          spacedim              = 2;
    std::string  input_parameter_file  = "";
    std::string  output_parameter_file = "";

    const std::string exename =
      cli(0).str().substr(cli(0).str().find_last_of("/") + 1);


    // Either as -i prm_file or as first positional argument after all options
    cli({"i", "input_prm_file"}, input_parameter_file) >> input_parameter_file;
    cli(1, input_parameter_file) >> input_parameter_file;

    // Now the logic to deduce dim and spacedim from the parameter file name
    bool file_contains_dim_spacedim = false;
    if (input_parameter_file.find("1d_2d") != std::string::npos)
      {
        dim                        = 1;
        spacedim                   = 2;
        file_contains_dim_spacedim = true;
      }
    else if (input_parameter_file.find("1d_3d") != std::string::npos)
      {
        dim                        = 1;
        spacedim                   = 3;
        file_contains_dim_spacedim = true;
      }
    else if (input_parameter_file.find("2d_3d") != std::string::npos)
      {
        dim                        = 2;
        spacedim                   = 3;
        file_contains_dim_spacedim = true;
      }
    else if (input_parameter_file.find("1d") != std::string::npos)
      {
        dim                        = 1;
        spacedim                   = 1;
        file_contains_dim_spacedim = true;
      }
    else if (input_parameter_file.find("2d") != std::string::npos)
      {
        dim                        = 2;
        spacedim                   = 2;
        file_contains_dim_spacedim = true;
      }
    else if (input_parameter_file.find("3d") != std::string::npos)
      {
        dim                        = 3;
        spacedim                   = 3;
        file_contains_dim_spacedim = true;
      }

    // Make sure filename and command line arguments agree
    if (file_contains_dim_spacedim == true)
      {
        int cli_dim      = dim;
        int cli_spacedim = spacedim;
        if ((cli({"d", "dim"}) >> cli_dim) ||
            (cli({"s", "spacedim"}) >> cli_spacedim))
          {
            AssertThrow(
              (dim == cli_dim) && (spacedim == cli_spacedim),
              dealii::ExcMessage(
                "You have specified a parameter filename that contains a "
                "specification of the dimension and of the space dimension, "
                "e.g., 1d_2d but you also indicated a -d or -s argument on the "
                "command line that do not match the file name. Use only one of "
                "the two ways to select the dimension and the space dimension, "
                "or make sure that what you specify on the filename matches "
                "what you specify on the command line."));
          }
      }
    else
      {
        // get dim and spacedim from command line
        cli({"d", "dim"}, dim) >> dim;
        // By default spacedim is equal to dim
        cli({"s", "spacedim"}, dim) >> spacedim;
      }

    // Now the logic to deduce the output parameter file name. Make sure we
    // output in the current directory, even if the file is specified with a
    // full path
    if (input_parameter_file != "")
      {
        auto rel_name = input_parameter_file.substr(
          input_parameter_file.find_last_of("/") + 1);
        output_parameter_file = "used_" + rel_name;
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
    cli({"o", "output_prm_file"}, output_parameter_file) >>
      output_parameter_file;

    deallog << "Will run in dimension " << dim << " and spacedimemsion "
            << spacedim << std::endl
            << "Input parameter file: " << input_parameter_file << std::endl
            << "Output parameter file: " << output_parameter_file << std::endl;

    return std::make_tuple(dim,
                           spacedim,
                           input_parameter_file,
                           output_parameter_file);
  }


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