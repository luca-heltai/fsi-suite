#include "pdes/serial/distributed_lagrange.h"

using namespace dealii;

/**
 * Parallel distributed Lagrange finite element method.
 */
int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
      std::string par_name = "distributed_lagrange_2d_1d.prm";
      if (argc > 1)
        par_name = argv[1];
      PDEs::Serial::DistributedLagrange<1, 2> distributed_lagrange;
      ParameterAcceptor::initialize(par_name, "used_" + par_name);
      distributed_lagrange.run();
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
