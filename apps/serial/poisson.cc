#include "pdes/serial/poisson.h"

#include "parsed_tools/grid_generator.h"

using namespace dealii;

/**
 * Serial Poisson solver.
 */
int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
      std::string                      par_name = "poisson_parameters_2d.prm";
      if (argc > 1)
        par_name = argv[1];
      PDEs::Serial::Poisson<2> poisson;
      ParameterAcceptor::initialize(par_name, "used_" + par_name);
      poisson.run();
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
