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

/**
 * Gather some functions and classes typically used in the `main()` of the
 * FSI-suite applications.
 */
namespace Runner
{
  /**
   * Parse from the command line the parameter file names (both input and
   * output) and the running dimensions (both dim and spacedim).
   *
   * This function uses the command line interface parsers from the argh library
   * (https://github.com/adishavit/argh), and recognizes the following
   * parameters from the command line:
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
   * function setup_parameters_from_cli(). The order of execution must follow
   * the following guidelines:
   * 1. call get_dimensions_and_parameter_files()
   * 2. initialize the classes of the correct dimension and spacedimension
   * 3. call ParameterAcceptor::initialize()
   * 4. call setup_parameters_from_cli()
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
  get_dimensions_and_parameter_files(char **argv);

  /**
   * Setup the ParameterAcceptor::prm according to the parameters specified in
   * the parameter file, and the parameters specified from the command line.
   *
   * This function uses the command line interface parsers from the argh library
   * (https://github.com/adishavit/argh), and allows you to specify input
   * parameter files, output parameter files, and to change any option
   * recognized by the parameter file itself from the command line.
   *
   * This function is usually used in conjunction with the function
   * get_dimensions_and_parameter_files(), in order to ask from the command line
   * what simulation to run (1d, 2d, 3d, etc.), and what input parameter files
   * to read from. These are then passed to this function, **after** you have
   * called at least once ParameterAcceptor::initialize() on your classes (which
   * you will have instantiated with the correct dimension thanks to the
   * function above).
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
   * -i, --input_prm_file <filename>   Input parameter file. Defaults to the
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
   * followed by a list of the options that are recognized by the parameter
   * file, i.e., for the mesh_handler.cc application, we get:
   * @code {.sh}
   * ...
   * Listing of Parameters:
   *
   * set Arguments                 = Any string
   * set Initial grid refinement   = An integer
   * set Input name                = Any string
   * set Output name               = Any string
   * set Transform to simplex grid = A boolean value (true or false)
   * set Verbosity                 = An integer n such that -1<= n <=2147483647
   * @endcode
   *
   * @warning This function is usually called after classes are constructed and
   * initialized (i.e., after calling ParameterAcceptor::initialize()). In order
   * to extract dimension and spacedimension from the command line, you can use
   * this function in conjunction with its companion function
   * get_dimensions_and_parameter_files(). The order of execution must follow
   * the following guidelines:
   * 1. call get_dimensions_and_parameter_files()
   * 2. initialize the classes of the correct dimension and spacedimension
   * 3. call ParameterAcceptor::initialize()
   * 4. call setup_parameters_from_cli()
   *
   * @return 0: everything is fine, 1: unused parameters, -1: help printed
   */
  int
  setup_parameters_from_cli(char **            argv,
                            const std::string &input_parameter_file,
                            const std::string &output_parameter_file);

  /**
   * Setup parameters from the command line, and call the Class::run() method.
   *
   * @tparam Class Type of the class to instantiate and run.
   * @param argv Arguments of the command line.
   * @param input_parameter_file Input parameter file.
   * @param output_parameter_file Output parameter file.
   */
  template <typename Class>
  void
  run(char **            argv,
      const std::string &input_parameter_file,
      const std::string &output_parameter_file)
  {
    Class class_name;
    if (setup_parameters_from_cli(argv,
                                  input_parameter_file,
                                  output_parameter_file) == -1)
      return;
    class_name.run();
  }

  template <template <int, int> class Class>
  bool
  run_codim(char **argv)
  {
    // Get the dimension and spacedimension from the command line
    const auto [dim, spacedim, in_file, out_file] =
      get_dimensions_and_parameter_files(argv);

    if (dim == 1 && spacedim == 2)
      Runner::run<Class<1, 2>>(argv, in_file, out_file);
    else if (dim == 2 && spacedim == 3)
      Runner::run<Class<2, 3>>(argv, in_file, out_file);
    else
      return false;
    return true;
  }

  template <template <int> class Class>
  bool
  run_dim_noone(char **argv)
  {
    // Get the dimension and spacedimension from the command line
    const auto [dim, spacedim, in_file, out_file] =
      get_dimensions_and_parameter_files(argv);
    if (dim == 2 && spacedim == 2)
      Runner::run<Class<2>>(argv, in_file, out_file);
    else if (dim == 3 && spacedim == 3)
      Runner::run<Class<3>>(argv, in_file, out_file);
    else
      return false;
    return true;
  }

  template <template <int> class Class>
  bool
  run_dim(char **argv)
  {
    // Get the dimension and spacedimension from the command line
    const auto [dim, spacedim, in_file, out_file] =
      get_dimensions_and_parameter_files(argv);
    if (dim == 1 && spacedim == 2)
      Runner::run<Class<1>>(argv, in_file, out_file);
    else if (run_dim_noone<Class>(argv))
      {}
    else
      return false;
    return true;
  }
} // namespace Runner

#ifndef DOXYGEN

#  define RUN_CODIM(Class, dim, spacedim, in_file, out_file) \
    if (dim == 1 && spacedim == 2)                           \
      Runner::run<Class<1, 2>>(argv, in_file, out_file);     \
    else if (dim == 2 && spacedim == 3)                      \
      Runner::run<Class<2, 3>>(argv, in_file, out_file);

#  define RUN_DIM_NO_ONE(Class, dim, spacedim, in_file, out_file) \
    if (dim == 2 && spacedim == 2)                                \
      Runner::run<Class<2>>(argv, in_file, out_file);             \
    else if (dim == 3 && spacedim == 3)                           \
      Runner::run<Class<3>>(argv, in_file, out_file);


#  define RUN_DIM(Class, dim, spacedim, in_file, out_file) \
    if (dim == 1 && spacedim == 1)                         \
      Runner::run<Class<1>>(argv, in_file, out_file);      \
    else                                                   \
      RUN_DIM_NO_ONE(Class, dim, spacedim, in_file, out_file)

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
#define RUNNER_DIM(Class, argc, argv)                                          \
  try                                                                          \
    {                                                                          \
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv); \
      const auto [dim, spacedim, in_file, out_file] =                          \
        Runner::get_dimensions_and_parameter_files(argv);                      \
      RUN_DIM(Class, dim, spacedim, in_file, out_file)                         \
      else AssertThrow(false,                                                  \
                       dealii::ExcImpossibleInDimSpacedim(dim, spacedim));     \
    }                                                                          \
  STANDARD_CATCH()

/**
 * Body of the main function for a program that runs a simulation in dimension
 * dim only using the ParameterAcceptor class, dimensions 1, 2, and 3.
 */
#define RUNNER_DIM_NO_ONE(Class, argc, argv)                                   \
  try                                                                          \
    {                                                                          \
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv); \
      const auto [dim, spacedim, in_file, out_file] =                          \
        Runner::get_dimensions_and_parameter_files(argv);                      \
      RUN_DIM_NO_ONE(Class, dim, spacedim, in_file, out_file)                  \
      else AssertThrow(false,                                                  \
                       dealii::ExcImpossibleInDimSpacedim(dim, spacedim));     \
    }                                                                          \
  STANDARD_CATCH()


/**
 * Body of the main function for a program that runs a simulation in dimension
 * dim only using the ParameterAcceptor class, dimensions 1, 2, and 3.
 */
#define RUNNER(Class, argc, argv)                                              \
  try                                                                          \
    {                                                                          \
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv); \
      const auto [dim, spacedim, in_file, out_file] =                          \
        Runner::get_dimensions_and_parameter_files(argv);                      \
      RUN_DIM(Class, dim, spacedim, in_file, out_file)                         \
      else RUN_CODIM(                                                          \
        Class,                                                                 \
        dim,                                                                   \
        spacedim,                                                              \
        in_file,                                                               \
        out_file) else AssertThrow(false,                                      \
                                   dealii::ExcImpossibleInDimSpacedim(         \
                                     dim, spacedim));                          \
    }                                                                          \
  STANDARD_CATCH()


/**
 * Body of the main function for a program that runs a simulation in dimensions
 * two or three using the ParameterAcceptor class. Not instantiated for
 * dimension one.
 */
#define RUNNER_NO_ONE(Class, argc, argv)                                       \
  try                                                                          \
    {                                                                          \
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv); \
      const auto [dim, spacedim, in_file, out_file] =                          \
        Runner::get_dimensions_and_parameter_files(argv);                      \
      RUN_DIM_NO_ONE(Class, dim, spacedim, in_file, out_file)                  \
      else RUN_CODIM(                                                          \
        Class,                                                                 \
        dim,                                                                   \
        spacedim,                                                              \
        in_file,                                                               \
        out_file) else AssertThrow(false,                                      \
                                   dealii::ExcImpossibleInDimSpacedim(         \
                                     dim, spacedim));                          \
    }                                                                          \
  STANDARD_CATCH()
#endif