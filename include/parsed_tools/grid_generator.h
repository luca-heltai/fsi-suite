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
   * function GridGenerator::generate_from_name_and_arguments(), and for the
   * classes GridIn and GridOut.
   *
   * Example usage:
   * @code
   * GridGenerator<dim, spacedim> pgg("/Grid");
   * Triangulation<dim,spacedim> tria;
   *
   * ParameterAcceptor::initialize("parameters.prm");
   *
   * pgg.generate(tria);
   * pgg.write(tria);
   * @endcode
   *
   * See the documentation of the ParameterAcceptor class for more information
   * on how parameter files and section names are handled.
   *
   * A parameter file that would work with the example above is:
   * @code{.sh}
   * subsection Grid
   *   set Input name                = hyper_cube
   *   set Arguments                 = 0: 1: false
   *   set Initial grid refinement   = 1
   *   set Output name               = grid_out.msh
   *   set Transform to simplex grid = false
   * end
   * @endcode
   *
   * The above example allows you to generate() a hypercube with side 1,
   * lower left corner at the origin, and with all boundary ids set to zero by
   * default. See GridGenerator::hyper_cube() for an explanation of all the
   * arguments.
   *
   * The coarse grid is written to the file `grid_out.msh`, and the grid is
   * refined once after beeing generated and written to `grid_out.msh`.
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
    GridGenerator(const std::string &prm_section_path = "");

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
          const std::string &                         filename = "") const;

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
    std::string grid_generator_function = "hyper_cube";

    /**
     * Arguments to the grid generator function. See the documentation of
     * GridGenerator::generate_from_name_and_arguments() for a description of
     * how to format the input string.
     */
    std::string grid_generator_arguments = "0: 1: false";

    /** Name of the output file. */
    std::string output_file_name = "";

    /** Transform quad and hex grids to simplex grids. */
    bool transform_to_simplex_grid = false;

    /** Initial global refinement of the grid. */
    unsigned int initial_grid_refinement = 0;
  };
} // namespace ParsedTools

#endif