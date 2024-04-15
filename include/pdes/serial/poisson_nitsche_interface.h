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


#ifndef pdes_serial_poisson_nitsche_h
#define pdes_serial_poisson_nitsche_h

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
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

#include "assemble_nitsche_with_exact_intersections.h"
#include "create_coupling_sparsity_pattern_with_exact_intersections.h"
#include "create_nitsche_rhs_with_exact_intersections.h"
#include "parsed_lac/amg.h"
#include "parsed_lac/inverse_operator.h"
#include "parsed_tools/boundary_conditions.h"
#include "parsed_tools/constants.h"
#include "parsed_tools/convergence_table.h"
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
     * Imposing an interaface condition in Poisson problem, serial version.
     *
     * @ingroup basics
     * Here we solve the Poisson equation:
     * \f[
     * \begin{cases}
     *  - \Delta u = f & \text{ in } \Omega \subset R^{\text{spacedim}}\\
     *    u = u_D = 0 & \text{ on } \partial \Omega \\
     *    u = g & \text{on} B
     *    \\
     * \end{cases}
     * \f]
     *
     * where B is a domain *embedded* in \Omega. The structure of the program is
     * the usual one, what really changes is the weak form. To get that, we
     * apply Nitsche's method in $\Omega \setminus B$ and $B$, separately, as
     * done in step-70 tutorial. For an extended discussion see there and
     * references therein. All in all, what one gets using Lagrangian finite
     * elements in codimension 1 is to find $u \in H_0^1(\Omega)$ s.t.
     *
     * @f{align*} (\nabla u, \nabla v) + 2 \beta \langle u,v \rangle_{\Gamma}
     * =(f,v) + 2 \beta \langle g,v \rangle_{\Gamma}
     * @f}
     * for all test functions $v \in H_0^1(\Omega)$. Here $\Gamma
     * = \partial B$ in codimension 1. The contributions from Nitsche's method
     * then have to be added both to the stiffness matrix and to the right hand
     * side. The thing here is that they have to be computed on the embedded
     * grid, while $u$ and $v$ live on the ambient space.
     */

    template <int dim, int spacedim = dim>
    class PoissonNitscheInterface : public ParameterAcceptor
    {
    public:
      PoissonNitscheInterface();


      void
      run();

    protected:
      void
      generate_grids();


      /**
       * Setup dofs, constraints, and matrices.
       *
       * This function enumerate all the degrees of freedom and set up matrix
       * and vector objects to hold the system data. Enumerating is done by
       * using DoFHandler::distribute_dofs(), as we have seen in the step-2
       * example, with the difference that now we pass a
       * ParsedTools::FiniteElement object instead of directly a
       * dealii::FiniteElement object.
       */
      void

      setup_system();


      void
      assemble_system();


      void
      solve();


      void
      output_results(const unsigned cycle) const;


      const std::string component_names = "u";


      ParsedTools::GridGenerator<spacedim, spacedim> grid_generator;

      ParsedTools::GridGenerator<dim, spacedim> embedded_grid_generator;

      ParsedTools::GridRefinement grid_refinement;

      /**
       * The actual triangulations. Here with "space_triangulation" we refer to
       * the original domain \Omega, also called the ambient space, while with
       * embedded we refer to the immersed domain, the one where we want to
       * impose a constraint.
       *
       */
      Triangulation<spacedim, spacedim> space_triangulation;
      Triangulation<dim, spacedim>      embedded_triangulation;

      /**
       * GridTools::Cache objects are used to cache all the necessary
       * information about a given triangulation, such as its Mapping, Bounding
       * Boxes, etc.
       *
       */
      std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
      std::unique_ptr<GridTools::Cache<dim, spacedim>>      embedded_cache;

      /**
       * The coupling between the two grids is ultimately encoded in this
       * vector. Here the i-th entry stores a tuple for which the first two
       * elements are iterators to two cells from the space and embedded grid,
       * respectively, that intersect each other (up to a specified tolerance)
       * and a Quadrature object to integrate over that region.
       *
       *
       */
      std::vector<std::tuple<
        typename dealii::Triangulation<spacedim, spacedim>::cell_iterator,
        typename dealii::Triangulation<dim, spacedim>::cell_iterator,
        dealii::Quadrature<spacedim>>>
        cells_and_quads;


      ParsedTools::FiniteElement<spacedim, spacedim> space_fe;

      /**
       * The actual DoFHandler class.
       */
      DoFHandler<spacedim, spacedim> space_dh;

      /**
       * According to the Triangulation type, we use a MappingFE or a MappingQ,
       * to make sure we can run the program both on a tria/tetra grid and on
       * quad/hex grids.
       */
      std::unique_ptr<Mapping<spacedim, spacedim>> mapping;
      /** @} */

      /**
       * \name Linear algebra classes
       * @{
       */
      AffineConstraints<double>    space_constraints;
      SparsityPattern              sparsity_pattern;
      SparseMatrix<double>         system_matrix;
      Vector<double>               solution;
      Vector<double>               system_rhs;
      ParsedLAC::InverseOperator   inverse_operator;
      ParsedLAC::AMGPreconditioner preconditioner;
      /** @} */

      /**
       * \name Forcing terms and boundary conditions
       * @{
       */


      ParsedTools::Constants constants;

      /**
       * The actual function to use as a forcing term.
       */
      ParsedTools::Function<spacedim> forcing_term;


      /**
       * This is the value we want to impose on the embedded domain.
       *
       */
      ParsedTools::Function<spacedim> embedded_value;


      /**
       * The coefficient in front of the Nitsche contribution to the stiffness
       * matrix.
       *
       */
      ParsedTools::Function<spacedim> nitsche_coefficient;

      /**
       * The actual function to use as a exact solution when computing the
       * errors.
       * */
      ParsedTools::Function<spacedim> exact_solution;


      ParsedTools::BoundaryConditions<spacedim> boundary_conditions;

      /**
       * \name Output and postprocessing
       * @{
       */

      mutable TimerOutput timer;


      ParsedTools::ConvergenceTable error_table;



      /**
       * Choosing as embedded space the square $[-.0.45,0.45]^2$ and as
       * embedding space the square $[-1,1]^2$, with embedded value the
       * function $g(x,y)=1$, this is what we get
       * @image html Poisson_1_interface.png
       *
       *
       * Taking a manufactured smooth solution $u=\sin(2 \pi x) \sin(2 \pi y)$,
       * classical rates can be observed, as in the following table:
       * cells dofs   u_L2_norm    u_Linfty_norm    u_H1_norm
         256  289 5.851e-02    - 8.125e-02    - 2.015e+00    -
        1024 1089 1.436e-02 2.12 2.160e-02 2.00 1.007e+00 1.05
        4096 4225 3.605e-03 2.04 5.519e-03 2.01 5.037e-01 1.02
       */
      mutable ParsedTools::DataOut<spacedim, spacedim> data_out;

      /**
       * Level of log verbosity.

       */
      unsigned int console_level = 1;



      /**
       * The penalty parameter which multiplies Nitsche's terms. In this program
       * it is defaulted to 100.0
       *
       */

      double penalty = 100.0;



      /** @} */
    };
  } // namespace Serial
} // namespace PDEs
#endif
