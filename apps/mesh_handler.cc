#include "tools/parsed_grid_generator.h"

/**
 * Generate a mesh reader.
 */
int
main(int argc, char **argv)
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      par_name = "";
      if (argc > 1)
        par_name = argv[1];

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      Tools::ParsedGridGenerator<2> pgg;
      ParameterAcceptor::initialize(par_name);
      Triangulation<2> tria;
      pgg.generate(tria);
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
