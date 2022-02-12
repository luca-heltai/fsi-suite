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

/**
 * Degrees of freedom plotter.
 *
 * @file dof_plotter.cc
 * @ingroup basics
 *
 * Illustrates the use of ParsedTools::GridGenerator,
 * ParsedTools::FiniteElement, and ParsedTools::DataOut. It surrogates step-2 of
 * the deal.II library, and it builds on mesh_handler.cc.
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>

#include "parsed_tools/components.h"
#include "parsed_tools/data_out.h"
#include "parsed_tools/finite_element.h"
#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_info.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      par_name = "dof_plotter.prm";
      if (argc > 1)
        par_name = argv[1];

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      ParsedTools::GridGenerator<2> pgg;
      ParsedTools::FiniteElement<2> pfe;
      ParsedTools::DataOut<2>       pdo;
      unsigned int                  n_basis_functions = 3;
      unsigned int                  info_level        = 0;
      unsigned int                  mapping_degree    = 1;


      ParameterAcceptor::prm.add_parameter(
        "Maximum number of basis functions to plot", n_basis_functions);
      ParameterAcceptor::prm.add_parameter("Mapping degree", mapping_degree);
      ParameterAcceptor::prm.add_parameter("Verbosity of grid info",
                                           info_level);

      ParameterAcceptor::initialize(par_name, "used_" + par_name);

      Triangulation<2> tria;
      pgg.generate(tria);
      ParsedTools::GridInfo info(tria, info_level);
      pgg.write(tria);
      info.print_info(deallog);

      DoFHandler<2> dh(tria);
      dh.distribute_dofs(pfe);
      std::unique_ptr<Mapping<2>> mapping;
      if (tria.all_reference_cells_are_hyper_cube())
        mapping = std::make_unique<MappingQ<2>>(mapping_degree);
      else
        {
          mapping =
            std::make_unique<MappingFE<2>>(FE_SimplexP<2>(mapping_degree));
          //   tria.set_all_manifold_ids(1);
        }

      auto quad =
        ParsedTools::Components::get_cell_quadrature(tria,
                                                     pfe().tensor_degree() + 1);


      DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
      DoFTools::make_sparsity_pattern(dh, dsp);
      dsp.compress();
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);
      sparsity_pattern.compress();
      SparseMatrix<double> mass_matrix(sparsity_pattern);

      MatrixCreator::create_mass_matrix(*mapping, dh, quad, mass_matrix);

      SparseDirectUMFPACK mass_matrix_solver;
      mass_matrix_solver.initialize(mass_matrix);

      std::vector<Vector<double>> basis_functions(n_basis_functions,
                                                  Vector<double>(dh.n_dofs()));
      std::vector<Vector<double>> reciprocal_basis_functions(
        n_basis_functions, Vector<double>(dh.n_dofs()));

      pdo.attach_dof_handler(dh);
      n_basis_functions = std::min(n_basis_functions, dh.n_dofs());

      for (unsigned int i = 0; i < n_basis_functions; ++i)
        {
          basis_functions[i][i] = 1.0;
          mass_matrix_solver.vmult(reciprocal_basis_functions[i],
                                   basis_functions[i]);
          pdo.add_data_vector(basis_functions[i],
                              "basis_function_" + std::to_string(i));
          pdo.add_data_vector(reciprocal_basis_functions[i],
                              "reciprocal_basis_function_" + std::to_string(i));
        }
      pdo.write_data_and_clear(*mapping);
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
