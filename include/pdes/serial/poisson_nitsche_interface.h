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
     * (\nabla u, \nabla v) + 2 \beta <u,v>_{\Gamma} = (f,v) + 2 \beta
     * <g,v>_{\Gamma} for all test functions $v \in H_0^1(\Omega)$. Here $\Gamma
     * = \partial B$. The contributions from Nitsche's method then have to be
     * added both to the stiffness matrix and to the right hand side. The thing
     * here is that they have to be computed on the embedded grid, while $u$ and
     * $v$ live on the ambient space.
     */

    template <int dim, int spacedim = dim>
    class PoissonNitscheInterface : public ParameterAcceptor
    {
    public:
      /**
       * At construction time, we need to initialize all member functions that
       * are derived from ParameterAcceptor.
       *
       * The general design principle of each of these classes is to take as
       * first argument the name of section where you want store the parameters
       * for the given class, and then the default values that you want to store
       * in the parameter file when you create it the first time.
       */
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

      /**
       * How we identify the component names.
       *
       * Many of the classes in the ParsedTools namespace use a string to define
       * their behaviour in terms of the components of a problem. For example,
       * for a scalar problem, one usually needs only to specify the name of the
       * solution with a single component. This is what happens in this program:
       * we use "u" as an identification for the name of the solution, and pass
       * this string around to any class that needs information about
       * components. In general our codes in the FSI-suite will use the
       * functions defined in the ParsedTools::Components namespace to process
       * information about components.
       *
       * In the scalar case of this program, we have only one component, so this
       * string is not very informative, however, in a more complex case like
       * the PDEs::Serial::Stokes case, we would have spacedim+1 components,
       * spacedim components for the velocity and one component for the
       * pressure. In that case, this string would be "u,u,p" in the two
       * dimensional case, and "u,u,u,p" in the three dimensional case. This
       * string gives information about how we want to group variables, and how
       * we want to treat them in the output as well as how many components our
       * finite element spaces must have.
       */
      const std::string component_names = "u";


      ParsedTools::GridGenerator<dim, spacedim> grid_generator;

      ParsedTools::GridGenerator<dim, spacedim> embedded_grid_generator;

      ParsedTools::GridRefinement grid_refinement;

      /**
       * The actual triangulations. Here with "space_triangulation" we refer to
       * the original domain \Omega, also called the ambient space, while with
       * embedded we refer to the immersed domain, the one where we want to
       * impose a constraint.
       *
       */
      Triangulation<dim, spacedim> space_triangulation;
      Triangulation<dim, spacedim> embedded_triangulation;

      /**
       * GridTools::Cache objects are used to cache all the necessary
       * information about a given triangulation, such as its Mapping, Bounding
       * Boxes, etc.
       *
       */
      std::unique_ptr<GridTools::Cache<dim, spacedim>> space_cache;
      std::unique_ptr<GridTools::Cache<dim, spacedim>> embedded_cache;

      /**
       * The coupling between the two grids is ultimately encoded in this
       * vector. Here the i-th entry stores a tuple for which the first two
       * elements are iterators to two cells from the space and embedded grid,
       * respectively, that intersect each other (up to a specified tolerance)
       * and a Quadrature object to integrate over that region.
       *
       *
       */
      std::vector<
        std::tuple<typename dealii::Triangulation<dim, spacedim>::cell_iterator,
                   typename dealii::Triangulation<dim, spacedim>::cell_iterator,
                   dealii::Quadrature<spacedim>>>
        cells_and_quads;


      ParsedTools::FiniteElement<dim, spacedim> space_fe;

      /**
       * The actual DoFHandler class.
       */
      DoFHandler<dim, spacedim> space_dh;

      /**
       * According to the Triangulation type, we use a MappingFE or a MappingQ,
       * to make sure we can run the program both on a tria/tetra grid and on
       * quad/hex grids.
       */
      std::unique_ptr<Mapping<dim, spacedim>> mapping;
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
       * The actual function to use as a forcing term. This is a wrapper around
       * the dealii::ParsedFunction class, which allows you to define a function
       * through a symbolic expression (a string) in a parameter file.
       *
       * The action of this class is driven by the section `Functions`, with the
       * parameter `Forcing term`:
       * @code{.sh}
       * subsection Functions
       *  set Forcing term = kappa*8*PI^2*sin(2*PI*x)*sin(2*PI*y)
       * end
       * @endcode
       *
       * You can use any of the numerical constants that are defined in the
       * dealii::numbers namespace, such as PI, E, etc, as well as the constants
       * defined at construction time in the ParsedTools::Constants class.
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
       * errors. This is a wrapper around the dealii::ParsedFunction class,
       * which allows you to define a function through a symbolic expression (a
       * string) in a parameter file.
       *
       * The action of this class is driven by the section `Functions`, with the
       * parameter `Exact solution`:
       * @code{.sh}
       * subsection Functions
       *  set Exact solution = sin(2*PI*x)*sin(2*PI*y)
       * end
       * @endcode
       *
       * You can use any of the numerical constants that are defined in the
       * dealii::numbers namespace, such as PI, E, etc, as well as the constants
       * defined at construction time in the ParsedTools::Constants class.
       */
      ParsedTools::Function<spacedim> exact_solution;

      /**
       * Boundary conditions used in this class.
       *
       * The action of this class is driven by the section `Boundary conditions`
       * of the parameter file:
       * @code{.sh}
       * subsection Boundary conditions
       *   set Boundary condition types (u) = dirichlet
       *   set Boundary id sets (u)         = -1
       *   set Expressions (u)              = 0
       *   set Selected components (u)      = u
       * end
       * @endcode
       *
       * The way ParsedTools::BoundaryConditions works in the FSI-suite is the
       * following: for every set of boundary ids of the triangulation, you need
       * to specify what boundary conditions are assumed to be imposed on that
       * set. If you only want to specify one type of boundary condition
       * (`dirichlet` or `neumann`) on all of the boundary, you can do so by
       * specifying `-1` as the boundary id set.
       *
       * Multiple boundary conditions can be specified, but the same id should
       * should appear only once in the parameter file (i.e., you cannot apply
       * different types of boundary conditions on the same boundary id).
       *
       * Keep in mind the following caveats:
       * - Boundary conditions are specified as comma separated strings, so you
       *   can specify "set Boundary condition types (u) = neumann, dirichlet"
       *   for two different sets of boundary ids.
       * - Following the previous example, different boundary id sets are
       *   separated by a semicolumn, and in each set, different boundary ids
       *   are separated by a column, so, for example, if you specify as
       *   `set Boundary id sets (u) = 0, 1; 2, 3`, then boundary ids 0 and 1
       *   will get Neumann boundary conditions, while boundary ids 2 and 3 will
       *   get Dirichlet boundary conditions.
       * - Since an expression can contain a `,` character, then expression for
       *   each component are separated by a semicolumn, and for each boundary
       *   id set, are separated by the `%` character. For example, if you want
       *   to specify homogeneous Neumann boundary conditions, and constant
       *   Dirichlet boundary conditions you can set the following parameter:
       *   `set Expressions (u) = 0 % 1`.
       * - The selected components, again can be `all`, a component name, or
       *   `u.n`, or `u.t` to select normal component, or tangential component
       *   in a vector valued problem. For scalar problems, only the name of the
       *   component makes sense. This field allows you to control which
       *   components the given boundary condition refers to.
       *
       * To summarize, the following is a valid section for the example above:
       * @code{.sh}
       * subsection Boundary conditions
       *   set Boundary condition types (u) = dirichlet, neumann
       *   set Boundary id sets (u)         = 0, 1 ; 2, 3
       *   set Expressions (u)              = 0 % 1
       *   set Selected components (u)      = u; u
       * end
       * @endcode
       *
       * This would apply Dirichlet boundary conditions on the boundary ids 2
       * and 3, and homogeneous Neumann boundary conditions on the boundary ids
       * 0 and 1.
       */
      ParsedTools::BoundaryConditions<spacedim> boundary_conditions;
      /** @} */

      /**
       * \name Output and postprocessing
       * @{
       */


      ParsedTools::ConvergenceTable error_table;



      /**
       * Choosing as embedded space the square $[-.0.45,0.45]^2$ and as
       * embedding space the square $[-1,1]^2$, with embedded value the
       * function $g(x,y)=1$, this is what we get
       * @image html Poisson_1_interface.png
       *
       */
      mutable ParsedTools::DataOut<dim, spacedim> data_out;

      /**
       * Level of log verbosity.
       *
       * This is the only "native" parameter of this class. All other parameters
       * are set through the constructors of the classes that inherit from
       * ParameterAcceptor.
       *
       * The console_level is used to setup the dealii::LogStream class, and
       * allows dealii clases to print messages to the console at different
       * level of detail and verbosity.
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
