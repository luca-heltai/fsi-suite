#include <deal.II/grid/reference_cell.h>

#include "tools/grid_info.h"
#include "tools/parsed_grid_generator.h"

using namespace dealii;

/**
 * Mesh generator and reader.
 *
 * This program is useful to debug input grid files. It gathers information from
 * the file specified in the input parameter and prints it on screen.
 */
int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      par_name   = "";
      unsigned int                     info_level = 0;
      if (argc > 1)
        par_name = argv[1];
      if (argc > 2)
        info_level = std::atoi(argv[2]);


      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      Tools::ParsedGridGenerator<2> pgg;
      ParameterAcceptor::initialize(par_name);
      Triangulation<2> tria;
      pgg.generate(tria);
      pgg.write(tria);
      Tools::GridInfo info(tria, info_level);
      info.print_info(deallog);
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
