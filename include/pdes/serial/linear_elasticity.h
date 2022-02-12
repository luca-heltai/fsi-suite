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

#ifndef pdes_serial_linear_elasticity_h
#define pdes_serial_linear_elasticity_h

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "parsed_lac/amg.h"
#include "parsed_lac/inverse_operator.h"
#include "parsed_tools/boundary_conditions.h"
#include "parsed_tools/constants.h"
#include "parsed_tools/data_out.h"
#include "parsed_tools/finite_element.h"
#include "parsed_tools/function.h"
#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_refinement.h"

namespace PDEs
{
  using namespace dealii;

  namespace Serial
  {
    /**
     * Serial LinearElasticity problem.
     *
     * @addtogroup csd
     *
     * Solve the LinearElasticity equation in arbitrary dimensions and space
     * dimensions. When dim and spacedim are not the same, we solve the
     * vector Laplace-Beltrami equation.
     *
     * \f[
     * \begin{cases}
     *  - \Delta u = f & \text{ in } \Omega \subset R^{\text{spacedim}}\\
     * u = u_D & \text{ on } \partial \Omega_D \\
     * \frac{\partial u}{\partial n} u = u_N & \text{ on } \partial \Omega_N \\
     * \frac{\partial u}{\partial n} u + \rho u= u_R & \text{ on } \partial
     * \Omega_R
     * \end{cases}
     * \f]
     *
     * @ingroup PDEs
     */
    template <int dim, int spacedim = dim>
    class LinearElasticity : public ParameterAcceptor
    {
    public:
      LinearElasticity();

      void
      run();

    protected:
      void
      setup_system();

      void
      assemble_system();

      void
      solve();

      void
      output_results(const unsigned cycle) const;

      /**
       * How we identify the component names.
       */
      const std::string component_names;

      // Grid classes
      ParsedTools::GridGenerator<dim, spacedim> grid_generator;
      ParsedTools::GridRefinement               grid_refinement;
      Triangulation<dim, spacedim>              triangulation;

      // FE and dofs classes
      ParsedTools::FiniteElement<dim, spacedim> finite_element;
      DoFHandler<dim, spacedim>                 dof_handler;
      std::unique_ptr<Mapping<dim, spacedim>>   mapping;

      // Linear algebra classes
      AffineConstraints<double>    constraints;
      SparsityPattern              sparsity_pattern;
      SparseMatrix<double>         system_matrix;
      Vector<double>               solution;
      Vector<double>               system_rhs;
      ParsedLAC::InverseOperator   inverse_operator;
      ParsedLAC::AMGPreconditioner preconditioner;

      // Forcing terms and boundary conditions
      ParsedTools::Constants                    constants;
      ParsedTools::Function<spacedim>           forcing_term;
      ParsedTools::Function<spacedim>           exact_solution;
      ParsedTools::BoundaryConditions<spacedim> boundary_conditions;

      // Error convergence tables
      ParsedConvergenceTable error_table;

      // Output class
      mutable ParsedTools::DataOut<dim, spacedim> data_out;

      // Console level
      unsigned int console_level = 1;

      // Extractor for vector components
      const FEValuesExtractors::Vector displacement;
    };
  } // namespace Serial
} // namespace PDEs
#endif
