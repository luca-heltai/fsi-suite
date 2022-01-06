#include "serial_poisson.h"

#include "tools/parsed_grid_generator.h"

using namespace dealii;

/**
 * Serial Poisson solver.
 */
int
main(int argc, char **argv)
{
  try
    {
      std::string par_name = "poisson_parameters_2d.prm";
      if (argc > 1)
        par_name = argv[1];
      PDEs::SerialPoisson<2> poisson;
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
