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

#ifndef parsed_tools_grid_generator_h
#define parsed_tools_grid_generator_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

namespace ParsedTools
{
  /**
   * GridGenerator class.
   *
   * This is an interface, derived from ParameterAcceptor, for the deal.II
   * function GridGenerator::generate_from_name_and_arguments(), for the classes
   * GridIn and GridOut, and for the OpenCASCADE
   * ArclengthProjectionLineManifold, NormalToMeshProjectionManifold, and
   * NURBSPatchManifold classes.
   *
   * Example usage:
   * @code
   * GridGenerator<dim, spacedim> pgg("/Grid"); Triangulation<dim,spacedim>
   * tria;
   *
   * ParameterAcceptor::initialize("parameters.prm");
   *
   * pgg.generate(tria); pgg.write(tria);
   * @endcode
   *
   * This class follows the design of the ParameterAcceptor class to handle
   * parameter files and section names.
   *
   * The default set of parameters usad to drive this class is given by:
   * @code{.sh} set Input name                = hyper_cube set Arguments
   * = 0: 1: false set Initial grid refinement   = 1 set Output name
   * = grid_out.msh set Transform to simplex grid = false
   * @endcode
   *
   * The above example allows you to generate() a hypercube with side 1, lower
   * left corner at the origin, and with all boundary ids set to zero by
   * default. See GridGenerator::hyper_cube() for an explanation of all the
   * arguments.
   *
   * ## Input name
   *
   * This class understands two types of `Input name` arguments:
   * 1. the name of one of the functions in the dealii::GridGenerator namespace
   *    of the deal.II library, as understood by the
   *    GridGenerator::generate_from_name_and_arguments() function;
   * 2. A file name, which is understood by the GridIn class of the deal.II
   *    library. In this case, the format of the name is deduced by the file
   *    extension. If everything else fails, GridIn::read_assimp() is used as a
   *    last resort.
   *
   * @warning If you have defined DEAL_II_WITH_MSH, and DEAL_II_GMSH_WITH_API,
   * the preferred way to read `.msh` files will be through the api version,
   * which supports the writing of both boundary ids, material ids and manifold
   * ids.
   *
   * ## Arguments
   *
   * When the `Input name` argument is a function in the deal.II library, the
   * `Arguments` parameter is interpreted as a list of arguments to be passed to
   * the function itself, and it will be passed, together with the `Input name`
   * parameter to the GridGenerator::generate_from_name_and_arguments()
   * function.
   *
   * If the argument is a filename, then the `Arguments` parameter is
   * interpreted as a map from manifold ids to CAD file names (`IGES`, `STEP`,
   * and `STL` formats are supported) that you can use to specify the
   * Geometrical description of your domain (see @cite
   * HeltaiBangerthKronbichler-2021 for more details of how this works in the
   * deal.II library).
   *
   * @warning When writing your CAD files, make sure that each CAD file contains
   * a single shape or compound for each of the topological entities you want to
   * describe. This class uses the following interpretation:
   * 1. wires/edges and lines are fed to an ArclengthProjectionLineManifold, and
   *    can be used as manifolds for edges, both in two and three dimensions;
   * 2. surfaces and faces when `spacedim` is equal to two are fed to a
   *    NURBSPatchManifold, and can be used as a manifold for cells in two
   *    dimensions;
   * 3. surfaces and faces when `spacedim` is equal to three are fed to a
   *    NormalToMeshProjectionManifold, and can be used as a manifold for faces
   *    or edges in three dimensions;
   *
   * ## Output name
   *
   * The `Output name` argument is interpreted as a file name. If the extension
   * is understood by GridOut, then the **coarse** triangulation is written to
   * this file as soon as it is generated, before any refinement occurs.
   *
   * @warning The newly generated grid is written within the generate()
   * function, before any refinement occurs. This allows you to write a grid to
   * file, and then use the same file to read it back in as a coarse mesh,
   * *irrespective* of the `Initial grid refinement` parameter. If you want to
   * output explicitly the refined version of the grid, you should call the
   * write() method with the grid you want to output.
   *
   * ## Transform to simplex grid
   *
   * If this is set to true, the class assumes you have read or generated a hex
   * grid, and want to transform it to a simplex grid, i.e. the
   * GridGenerator::convert_hypercube_to_simplex_mesh() is called with the grid
   * that has been generated using `Input name` and `Arguments`.
   *
   * ## Copy boundary to manifold ids
   *
   * If you have read the mesh from a file format that does not support manifold
   * ids, you can copy boundary ids to manifold ids using this option.
   *
   * ## Examples
   *
   * In the following examples we assume that the class was instantiated with
   * section name equal to "/". We report the example input parameter file, and
   * show a plot of the resulting grid **after** refinement, i.e., we call the
   * write() method ourselves. In order to regenerate the following examples,
   * just place this somwhere in your code:
   *
   * @code{.cpp} Triangulation<dim, spacedim> tria; GridGenerator<dim, spacedim>
   * pgg("/"); ParameterAcceptor::initialize("parameters.prm");
   * pgg.generate(tria); pgg.write(tria);
   * @endcode
   *
   * ### Hyper shell
   *
   * Parameter file:
   * @code{.sh} set Arguments                 = 0,0: .5: 1: 5: true set Initial
   * grid refinement   = 4 set Input name                = hyper_shell set
   * Output name               = hyper_shell.vtk set Transform to simplex grid =
   * false
   * @endcode
   *
   * This will generate a hyper shell with center in the point `0,0`, inner
   * radius `.5`, outer radius `1`, and with `5` cells the angular direction,
   * colorizing the boundary ids to 0 and 1. The output grid will look like the
   * following:
   * @image html hyper_shell.png and the **coarse** grid (before any refinement
   * takes place) will also be written to `hyper_shell.vtk` as soon as the grid
   * is generated.
   *
   * If you later call again the write() method with another Triangulation as
   * input parameter, then the argument of the write() method will overwrite the
   * `hyper_shell.vtk` file (or the file you specified in the parameters) with
   * the content of the grid.
   *
   * @ingroup grid
   */
  template <int dim, int spacedim = dim>
  class GridGenerator : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor. Initialize all parameters, and make sure the class is ready
     * to run.
     *
     * @param prm_section_path Name of the section to use in the parameter file.
     */
    GridGenerator(const std::string &prm_section_path          = "",
                  const std::string &grid_generator_function   = "hyper_cube",
                  const std::string &grid_generator_arguments  = "0: 1: false",
                  const std::string &output_file_name          = "",
                  const bool         transform_to_simplex_grid = false,
                  const unsigned int initial_grid_refinement   = 0,
                  const bool         copy_boundary_to_manifold_ids = false);

    /**
     * Fill a triangulation according to the parsed parameters. If the
     * @p output_file_name variable is not empty, the coarse triangulation is
     * also
     * saved to disk in the format specified by the the @p output_file_format.
     *
     * Notice that the triangulation is written to disk before any initial
     * refinement occurs. This allows you to store the Triangulation to a file,
     * and then use the same input file you used here with the exception of the
     * input/output grid names, and reproduce the same results.
     *
     * If the triangulation would be refined before output, running the same
     * program twice with input and output grid with the same name, would
     * produce more and more refined grids. If you really want to output the
     * same grid in the refined case, simply call the write() function again.
     */
    void
    generate(dealii::Triangulation<dim, spacedim> &tria) const;

    /**
     * Write the given Triangulation to the output file specified in `Output
     * file name`, or in the optional file name.
     *
     * If no `Output file name` is given and filename is the empty string,
     * this function does nothing. If an output file name is provided (either in
     * the input file, or as an argument to this function), then this function
     * will call the appropriate GridOut method according to the extension of
     * the file name.
     */
    void
    write(const dealii::Triangulation<dim, spacedim> &tria,
          const std::string                          &filename = "") const;

  private:
    /**
     * Name of the grid to generate.
     *
     * See the documentation of
     * GridGenerator::generate_from_name_and_arguments() for a description of
     * how to format the input string.
     *
     * If the name does not coincide with a function in the GridGenerator
     * namespace, the name is assumed to be a file name.
     */
    std::string grid_generator_function;

    /**
     * Arguments to the grid generator function. See the documentation of
     * GridGenerator::generate_from_name_and_arguments() for a description of
     * how to format the input string.
     */
    std::string grid_generator_arguments;

    /** Name of the output file. */
    std::string output_file_name;

    /** Transform quad and hex grids to simplex grids. */
    bool transform_to_simplex_grid;

    /** Copy boundary to manifold ids. */
    bool copy_boundary_to_manifold_ids;

    /** Initial global refinement of the grid. */
    unsigned int initial_grid_refinement;
  };
} // namespace ParsedTools

#endif