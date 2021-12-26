#ifndef parsed_grid_generator_h
#define parsed_grid_generator_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

namespace Tools
{
  /**
   * GridGenerator class
   */
  template <int dim, int spacedim = dim>
  class ParsedGridGenerator : dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor. Initialize all parameters, and make sure the class is ready
     * to run.
     */
    ParsedGridGenerator(const std::string &prm_section_path = "");

    /**
     * Fill a triangulation according to the parsed parameters.
     */
    void
    generate(dealii::Triangulation<dim, spacedim> &tria) const;

    /**
     * Write the given Triangulation to the output file specified in
     * `Output file name`, or in the optional file name.
     *
     * If no `Output file name` is given and filename is the empty string,
     * this function does nothing. If
     * an output file name is provided (either in the input file, or as an
     * argument to this function), then this function will call the appropriate
     * GridOut method according to the extension of the file name.
     */
    void
    write(const dealii::Triangulation<dim, spacedim> &tria,
          const std::string &                         filename = "") const;

  private:
    std::string  grid_generator_function   = "hyper_cube";
    std::string  grid_generator_arguments  = "0: 1: false";
    std::string  output_file_name          = "";
    bool         transform_to_simplex_grid = false;
    unsigned int initial_grid_refinement   = 0;
  };
} // namespace Tools

#endif