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

#ifndef fsi_mesh_handler_h
#define fsi_mesh_handler_h

#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_info.h"

/**
 * Entry point of the FSI-suite programs.
 *
 * @ingroup basics
 *
 * In general individual programs are as small as possible, and as clean as
 * possible. Each code solves a single problem. A lot of duplication is
 * unavoidable, but this has the advantage that it is easy to make incremental
 * steps, and understand the most difficult parts of the FSI suite by walking
 * through examples that increase in complexity, pretty much like it is done in
 * the deal.II tutorial programs.
 *
 * The structure of the FSI suite programs follows very closely deal.II tutorial
 * programs. In each program, I try to point the reader to the steps that are
 * required to understand what is going on. This program, for example, is the
 * FSI-suite equivalent of the deal.II tutorial program step-1.
 *
 * We show here how to run any of the FSI-suite programs, and how the wrappers
 * of the ParsedTools namespace are designed.
 *
 * In general each program is split in three parts: a header file, containing
 * the main class declaration (in the directory @ref include), a source file (in
 * the directory @ref include) containing the implementation of the main class,
 * and a main application file (in the directory @ref apps), containing only the
 * main function.
 *
 * This particular case is a bit different, since we only have one single
 * function in the class declaration, and we do not split it into a header and a
 * source file, but keep everything here.
 *
 * Each class is derived from ParameterAcceptor, and has a public constructor
 * that takes no arguments (or only arguments for which a default is specified).
 * The user entry point for each class is the method run(). The main functions
 * in the @ref apps folder (see the mesh_handler.cc file) take care of parsing
 * the command line, deducing from the command line (or from the name of the
 * paramter file) in what dimension and space dimension you want to run the
 * code, and then they instantiate the specific class for the specific dimension
 * and spacedimension combination, read parameters from the parameter file, and
 * execute the `run()` method of the class.
 *
 * The interface to all programs is uniform. Running any of the programs with an
 * `-h` or `--help` argument will return something like the following:
 * @code {.sh}
 * ./mesh_handler --help
 * Usage: ./mesh_handler [OPTIONS] [PRM FILE]
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
 *                                   `used_mesh_handler' followed by a
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
 *                                   should run. Defaults to same value of dim.
 * -"Section/option name"=<value>    Any of the options that you can specify
 *                                   in the parameter file. The format here
 *                                   is the following: -"Section/Subsection/
 *                                   option"="value,another value", where the
 *                                   quotes are required only if an otpion
 *                                   contains spaces, or if a value contains
 *                                   separators, like commas, columns, etc.
 * @endcode
 * followed by a list of parameters that you can change from the command line,
 * or from within a parameter file.
 *
 * For this class, we have the following options, which are printed after the
 * more general section described above:
 * @code {.sh}
 * Listing of Parameters:
 *
 * set Arguments                  = Any string
 * set Initial grid refinement    = An integer
 * set Input name                 = Any string
 * set Output name                = Any string
 * set Transform to simplex grid  = A boolean value (true or false)
 * set Verbosity                  = An integer n such that -1 <= n <= 2147483647
 * @endcode
 *
 * Every program in the FSI-suite can be executed by calling it with
 * arguments specifying the dimension and space dimension, or can deduce these
 * numbers from the naming scheme of your parameter file. For example:
 * @code{.sh}
 * ./mesh_handler -d=2 -s=3 -p=input.prm -o=output.prm
 * @endcode
 * will run the poisson problem in dimension 2 with spacedimension 3, reading
 * options from the `input.prm` file in the current directory, and writing all
 * used options in the `output.prm` file in the current directory.
 *
 * The same effect can be obtained by naming your parameter file in such a way
 * that the filename contains the combination of dimension and spacedimension,
 * or just one if they are the same. For example, the above example could have
 * been achieved similarly by executing it as:
 * @code{.sh}
 * ./mesh_handler input_2d_3d.prm
 * @endcode
 * and an equivalent example running in two dimensions
 * @code{.sh}
 * ./mesh_handler input_2d.prm
 * @endcode
 *
 * Notice how you can specify the input paramter file also as the first
 * positional arguement to the program, i.e., these are equivalent ways to run
 * the program with the same parameters:
 * @code{.sh}
 * ./mesh_handler input_2d.prm
 * ./mesh_handler -i input_2d.prm
 * ./mesh_handler --input_prm_file input_2d.prm
 * @endcode
 *
 * @warning If you specify the dimension (and/or the spacdimension) from the
 * command line, and provide a parameter file as input that is named with a
 * different combination, you will get an error message. Either do not name your
 * file with `1d_2d`, and use `-d 2` and `-s 2`, or make sure that they are
 * consistent:
 * @code{.sh}
 * ./mesh_handler -d=2 input_2d.prm
 * @endcode
 *
 * Notice that `input.prm` does not need to exist prior to executing the
 * program. If it does not exist, the program will create it for you, print an
 * error message, and exit. If you run again the program with the same
 * parameters, without changing the input file, it will run using all the
 * default values for the parameters.
 *
 * The file `input.prm` may be empty, or may specify just the parameters that
 * you want to change from their default value. If you specify a parameter in
 * the command line, it will override the value specified in the parameter file.
 *
 * The output file `output.prm` will contain all the parameters that were used
 * during the execution of the program, so that you can reproduce the results of
 * the last simulation by simply re-running the program with the file that was
 * just generated.
 *
 * Any parameter that you can change from the parameter files, can be changed
 * also from the command line. The opposite is also true, with the exception of
 * the dimension (`-d` or `-dim`), space dimension ('-s' or `-spacedim`), and
 * the input and output parameter files (`-i` and `-o`), which do not have, in
 * general, a correspondending entry in the parameter file itself.
 *
 * For parameters with spaces in their names and sections, you can surround the
 * option and/or the values by quotes.
 *
 * The MeshHandler class does the following things, in order:
 * 1. it generates a mesh using ParsedTools::GridGenerator, according to the
 * parameters `Input name` and `Arguments`. This parameters are passed as they
 * are to the deal.II function GridGenerator::generate_from_name_and_argument().
 * If this process fails, the `Input name` is interpreted as a grid file
 * name, which the ParsedTools::GridGenerator tries to open using one of the
 * file readers available in the library. In this case, the `Arguments` is
 * interpreted as a map from a manifold id to a CAD file in IGES or STEP
 * format, which is read into a OpenCascade::TopoDS_Shape objects, and used
 * to describe the geometry of the grid.
 * 2. it transforms the mesh to a simplex grid, if required by the parameter
 * 3. it refines the grid according to the parameters `Initial grid refinement`
 * 4. it output the mesh to a file, according to the parameters `Output name`,
 *    in a format which is deduced from the filename extension.
 * 5. it analyses the mesh and prints some statistics to the screen.
 *
 * Let's see a couple of examples for the `mesh_handler` application. For
 * example, running with the following command line:
 * @code {.sh}
 * ./mesh_handler --Arguments="0,0: .5: 1: 5: true"  \
 *                --"Initial grid refinement"="4" \
 *                --"Input name"="hyper_shell" \
 *                --"Output name"="hyper_shell.vtk" \
 *                --Verbosity=4
 *                -o shell.prm
 * @endcode
 * will generate the file `shell.prm`:
 * @code{.sh}
 * set Arguments                 = 0,0: .5: 1: 5: true
 * set Initial grid refinement   = 4
 * set Input name                = hyper_shell
 * set Output name               = hyper_shell.vtk
 * set Transform to simplex grid = false
 * set Verbosity                 = 4
 * @endcode
 * and the file `hyper_shell.vtk`, which looks like:
 * @image html hyper_shell.png
 * and will print the following output to the screen:
 * @code{.sh}
 * DEAL::=================
 * DEAL::Used parameters:
 * DEAL::=================
 * DEAL:parameters::Arguments: 0,0: .5: 1: 5: true
 * DEAL:parameters::Initial grid refinement: 4
 * DEAL:parameters::Input name: hyper_shell
 * DEAL:parameters::Output name: hyper_shell.vtk
 * DEAL:parameters::Transform to simplex grid: false
 * DEAL:parameters::Verbosity: 4
 * DEAL::=================
 * DEAL::Grid information:
 * DEAL::=================
 * DEAL::Active cells  : 1280
 * DEAL::Vertices      : 1360
 * DEAL::Used vertices : 1360
 * DEAL::Levels        : 5
 * DEAL::Active cells/level  : 0, 0, 0, 0, 1280
 * DEAL::Cells/level         : 5, 20, 80, 320, 1280
 * DEAL::Boundary ids         : 0, 1
 * DEAL::Manifold ids         : 0
 * DEAL::Material ids         : 0
 * DEAL::Reference cell types : 3
 * DEAL::Boundary id:n_faces         : 0:80, 1:80
 * DEAL::Material id:n_cells         : 0:1280
 * DEAL::Manifold id:n_faces         : 0:2640
 * DEAL::Manifold id:n_cells         : 0:1280
 * DEAL::Reference cell type:n_cells : 3:1280
 * @endcode
 *
 * @tparam dim
 * @tparam spacedim
 */
template <int dim, int spacedim = dim>
class MeshHandler : public dealii::ParameterAcceptor
{
public:
  /**
   * Construct a new Mesh Handler object.
   *
   * The only role of this constructor is to initialize correctly all member
   * classes that are derived from ParameterAcceptor, in this case only the
   * ParsedTools::GridGenerator class.
   */
  MeshHandler()
    : dealii::ParameterAcceptor("/")
    , pgg("/")
  {
    add_parameter("Verbosity", verbosity);
  }

  /**
   * Run the actual MeshHandler application.
   *
   * This function does the following things, in order:
   * 1. it generates a mesh using ParsedTools::GridGenerator, according to the
   * parameters `Input name` and `Arguments`. This parameters are passed as they
   * are to the deal.II function
   * GridGenerator::generate_from_name_and_argument(). If this process fails,
   * the `Input name` is interpreted as a grid file name, which the
   * ParsedTools::GridGenerator tries to open using one of the file readers
   * available in the library. In this case, the `Arguments` is interpreted as a
   * map from a manifold id to a CAD file in IGES or STEP format, which is read
   * into a OpenCascade::TopoDS_Shape objects, and used to describe the geometry
   * of the grid.
   * 2. it transforms the mesh to a simplex grid, if required by the parameter
   * 3. it refines the grid according to the parameters `Initial grid
   * refinement`
   * 4. it output the mesh to a file, according to the parameters `Output name`,
   *    in a format which is deduced from the filename extension.
   * 5. it analyses the mesh and prints some statistics to the screen using the
   *    class ParsedTools::GridInfo.
   */
  void
  run()
  {
    using namespace dealii;
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
  /**
   * Verbosity to use in the dealii::LogStream class.
   */
  unsigned int verbosity = 2;

  /**
   * The actual ParsedTools::GridGenerator object.
   */
  ParsedTools::GridGenerator<dim, spacedim> pgg;
};
#endif