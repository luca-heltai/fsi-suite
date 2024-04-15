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


#ifndef pdes_serial_poisson_h
#define pdes_serial_poisson_h

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
     * Poisson problem, serial version.
     *
     * @ingroup basics
     *
     * Solve the Poisson equation in arbitrary dimensions and space dimensions.
     * When dim and spacedim are not the same, we solve the Laplace-Beltrami
     * equation. This example is based on the deal.II tutorial step-1, step-2,
     * step-3, step-4, step-5, step-6, and step-38.
     *
     * The documentation is only a slight rewording of the deal.II tutorials
     * step-3, step-4, step-5, and step-6, where the main differences are
     * related to the classes in the ParsedTools namespace. Basic deal.II
     * classes, like the functions in the dealii::GridGenerator namespace, or
     * the dealii::FiniteElement class, are replaced with their ParsedTools
     * counterparts, i.e., ParsedTools::GridGenerator,
     * ParsedTools::FiniteElement, ParsedTools::Function, and
     * ParsedTools::DataOut.
     *
     * In this tutorial program, we solve the Poisson equation:
     * \f[
     * \begin{cases}
     *  - \Delta u = f & \text{ in } \Omega \subset R^{\text{spacedim}}\\
     *    u = u_D & \text{ on } \partial \Omega_D \\
     *    \frac{\partial u}{\partial n} = u_N & \text{ on } \partial \Omega_N
     *    \\
     * \end{cases}
     * \f]
     *
     * We will solve this equation on any grid that can be generated using the
     * ParsedTools::GridGenerator class. An example usage of that class was
     * given in the file MeshHandler.
     *
     * In this program, the right hand side $f$, the Dirichlet boundary
     * condition $u_D$, and the Neumann boundary condition $u_N$ will be read
     * from a parameter file.
     *
     * From the basics of the finite element method, we know the steps we need
     * to take to approximate the solution $u$ by a finite dimensional
     * approximation. Specifically, we first need to derive the weak form of the
     * equation above, which we obtain by multiplying the equation by a test
     * function $\varphi$ and integrating over the domain $\Omega$:
     *
     * @f{align*} -\int_\Omega \varphi \Delta u = \int_\Omega \varphi f.
     * @f}
     *
     * This can be integrated by parts:
     * @f{align*} \int_\Omega \nabla\varphi \cdot \nabla u
     * -\int_{\partial\Omega} \varphi \mathbf{n}\cdot \nabla u = \int_\Omega
     * \varphi f
     * @f}
     *
     * The test functions $\varphi$ are chosen so that they are zero on the
     * Dirichlet part of the boundary $\gamma_D$, while the Neumann boundary
     * condition is replaced naturally in the integration by parts, and shows up
     * on the right hand side as a boundary supported data.  The final weak form
     * reads:
     *
     * Given $f (H^1_{0,\Gamma_D} (\Omega))^*$, find $u \in H^1_{u_D,\Gamma_D}
     * (\Omega)$ such that
     *
     * @f{align*} (\nabla\varphi, \nabla u) = (\varphi, f) + \int_{\Gamma_N} u_N
     *   \varphi \qquad \forall \varphi \in H^1_{0,\Gamma_D} (\Omega)
     * @f}
     *
     * In finite elements, we seek an approximation $u_h(\mathbf x)= U^j
     * \varphi_j(\mathbf x)$ (sum is implied on the repeated indices), where the
     * $U^j$ are the unknown expansion coefficients we need to determine (the
     * "degrees of freedom" of this problem), and $\varphi_i(\mathbf x)$ are the
     * finite element shape functions we will use. To define these shape
     * functions, we need the following:
     *
     * - A mesh on which to define shape functions. We use the same technique
     *   described in the mesh_handler.cc program, i.e., a
     *   ParsedTools::GridGenerator object, that allows you to specify any of
     *   the built in deal.II meshes, or an external mesh file, generated using
     *   some external mesher.
     * - A finite element that describes the shape functions we want to use on
     *   the reference cell. In dof_plotter.cc, we had already used an object of
     *   type ParsedTools::FiniteElement, which allowed you to choose one of the
     *   finite element spaces supported by deal.II.
     * - A DoFHandler object that enumerates all the degrees of freedom on the
     *   mesh, taking the reference cell description the finite element object
     *   provides as the basis.
     * - A Mapping that tells how the shape functions on the real cell are
     *   obtained from the shape functions defined by the finite element class
     *   on the reference cell.
     *
     * Through these steps, we now have a set of functions $\varphi_i$, and we
     * can define the weak form of the discrete problem: Find a function $u_h$,
     * i.e., find the expansion coefficients $U^j$ mentioned above, so that
     * @f{align*} (\nabla\varphi_i, \nabla u_h) = (\varphi_i, f) +
     * \int_{\Gamma_N} u_N \varphi_i, \qquad\qquad
     *   i=0\ldots N-1.
     * @f}
     *
     * This equation can be rewritten as a linear system if you insert the
     * representation $u_h(\mathbf x)= U^j \varphi_j(\mathbf x)$ and then
     * observe that
     * @f{align*}{ (\nabla\varphi_i, \nabla u_h) &= \left(\nabla\varphi_i,
     *   \nabla \Bigl[\sum_j U^j \varphi_j\Bigr]\right)
     * \\
     *   &= \sum_j \left(\nabla\varphi_i, \nabla \left[U^j
     * \varphi_j\right]\right)
     * \\
     *   &= \sum_j \left(\nabla\varphi_i, \nabla \varphi_j \right) U^j.
     * @f}
     *
     * With this, the problem reads: Find a vector $U$ so that
     * @f{align*}{
     *  A_{ij} U^j = F_i,
     * @f}
     * where the matrix $A$ and the right hand side $F$ are defined as
     * @f{align*}
     * A_{ij} &= (\nabla\varphi_i, \nabla \varphi_j),
     *   \\
     *   F_i &= (\varphi_i, f) + \int_{\Gamma_N} u_N \varphi_i.
     * @f}
     *
     *
     * The final piece of this introduction is to mention that after a linear
     * system is obtained, it is solved using an iterative solver and then
     * postprocessed: we create an output file using the DataOut class that can
     * then be visualized using one of the common visualization programs.
     */
    template <int dim, int spacedim = dim>
    class Poisson : public ParameterAcceptor
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
      Poisson();

      /**
       * Main entry point of the program.
       *
       * Just as in all deal.II tutorial programs, the run() function is called
       * from the place where a PDEs::Serial::Poisson  object is created, and it
       * is the one that calls all the other functions in their proper order.
       * Encapsulating this operation into the `run()` function, rather than
       * calling all the other functions from `main()` of the application makes
       * sure that you can change how the *separation of concerns* within this
       * class is implemented. For example, if one of the functions becomes too
       * big, you can split it up into two, and the only places you have to be
       * concerned about changing as a consequence are within this very same
       * class, and not anywhere else.
       *
       * As mentioned above, you will see this general structure -- sometimes
       * with variants in spelling of the functions' names, but in essentially
       * this order of separation of functionality -- again in all of the
       * following tutorial programs.
       */
      void
      run();

      /**
       * Triangulation getter.
       *
       * @return Triangulation<dim>&
       */
      Triangulation<dim, spacedim> &
      get_triangulation()
      {
        return triangulation;
      }

    protected:
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

      /**
       * Assemble the matrix and right hand side vector.
       *
       * This function translates the problem in the finite dimensional space to
       * an equivalent linear algebra problmem . These are the main steps that
       * need to be taken into account for this translation:
       *
       * - The object for $A$ is of type SparseMatrix while those for $U$ and
       * $F$ are of type Vector. We will see in the program below what classes
       * are used to solve linear systems.
       * - We need a way to form the integrals. In the finite element method,
       * this is most commonly done using quadrature, i.e. the integrals are
       * replaced by a weighted sum over a set of *quadrature points* on each
       * cell. That is, we first split the integral over $\Omega$ into integrals
       * over all cells,
       *   @f{align*} A_{ij} &= (\nabla\varphi_i, \nabla \varphi_j) = \sum_{K
       * \in
       *     {\mathbb T}} \int_K \nabla\varphi_i \cdot \nabla \varphi_j,
       *     \\
       *     F_i &= (\varphi_i, f)
       *     = \sum_{K \in {\mathbb T}} \int_K \varphi_i f,
       *   @f}
       *   and then approximate each cell's contribution by quadrature:
       *   @f{align*}
       *   A^K_{ij} &= \int_K \nabla\varphi_i \cdot \nabla \varphi_j
       *   \approx \sum_q \nabla\varphi_i(\mathbf x^K_q) \cdot \nabla
       *   \varphi_j(\mathbf x^K_q) w_q^K,
       *     \\
       *     F^K_i &=
       *     \int_K \varphi_i f + \int_{\Gamma_N} u_N \varphi_i
       *     \approx
       *     \sum_q \varphi_i(\mathbf x^K_q) f(\mathbf x^K_q) w^K_q +
       *     \sum_q \varphi_i(\mathbf x^\Gamma_q) f(\mathbf
       * x^\Gamma_q)w^\Gamma_q,
       *   @f}
       *   where $\mathbb{T} \approx \Omega$ is a Triangulation approximating
       *   the domain, $\mathbf x^K_q$ is the $q$th quadrature point on cell
       * $K$, and $w^K_q$ the $q$th quadrature weight. There are different parts
       * to what is needed in doing this, and we will discuss them in turn next.
       * - First, we need a way to describe the location $\mathbf x_q^K$ of
       *   quadrature points and their weights $w^K_q$. They are usually mapped
       *   from the reference cell in the same way as shape functions, i.e.,
       *   implicitly using the MappingQ1 class or, if you explicitly say so,
       *   through one of the other classes derived from Mapping. The locations
       *   and weights on the reference cell are described by objects derived
       *   from the Quadrature base class. Typically, one chooses a quadrature
       *   formula (i.e. a set of points and weights) so that the quadrature
       *   exactly equals the integral in the matrix; this can be achieved
       *   because all factors in the integral are polynomial, and is done by
       *   Gaussian quadrature formulas, implemented in the QGauss class.
       * - We then need something that can help us evaluate $\varphi_i(\mathbf
       *   x^K_q)$ on cell $K$. This is what the FEValues class does: it takes a
       *   finite element objects to describe $\varphi$ on the reference cell, a
       *   quadrature object to describe the quadrature points and weights, and
       * a mapping object (or implicitly takes the MappingQ1 class) and provides
       *   values and derivatives of the shape functions on the real cell $K$ as
       *   well as all sorts of other information needed for integration, at the
       *   quadrature points located on $K$. The process of computing the matrix
       *   and right hand side as a sum over all cells (and then a sum over
       *   quadrature points) is usually called *assembling the linear system*,
       * or *assembly* for short, using the meaning of the word related to
       *   [assembly line](https://en.wikipedia.org/wiki/Assembly_line), meaning
       *   ["the act of putting together a set of pieces, fragments, or
       *   elements"](https://en.wiktionary.org/wiki/assembly).
       *
       * FEValues really is the central class in the assembly process. One way
       * you can view it is as follows: The FiniteElement and derived classes
       * describe shape <i>functions</i>, i.e., infinite dimensional objects:
       * functions have values at every point. We need this for theoretical
       * reasons because we want to perform our analysis with integrals over
       * functions. However, for a computer, this is a very difficult concept,
       * since they can in general only deal with a finite amount of
       * information, and so we replace integrals by sums over quadrature points
       * that we obtain by mapping (the Mapping object) using  points defined on
       * a reference cell (the Quadrature object) onto points on the real cell.
       * In essence, we reduce the problem to one where we only need a finite
       * amount of information, namely shape function values and derivatives,
       * quadrature weights, normal vectors, etc, exclusively at a finite set of
       * points. The FEValues class is the one that brings the three components
       * together and provides this finite set of information on a particular
       * cell $K$. You will see it in action when we assemble the linear system
       * below.
       *
       * It is noteworthy that all of this could also be achieved if you simply
       * created these three objects yourself in an application program, and
       * juggled the information yourself. However, this would neither be
       * simpler (the FEValues class provides exactly the kind of information
       * you actually need) nor faster: the FEValues class is highly optimized
       * to only compute on each cell the particular information you need; if
       * anything can be re-used from the previous cell, then it will do so, and
       * there is a lot of code in that class to make sure things are cached
       * wherever this is advantageous.
       */
      void
      assemble_system();

      /**
       * Solve the linear system generated in the assemble_system() function.
       *
       * After a linear system is obtained, it is solved using an iterative
       * solver. Differently from step-3, we would like to tweak the solver
       * parameters a bit. In order to do so, we use the
       * ParsedLAC::InverseOperator class and the ParsedLAC::AMGPreconditioner,
       * which are wrappers around the deal.II classes that are derived from
       * SolverBase, around the SolverControl class introduced in step-3 (the
       * ParsedLAC::InverseOperator) and around the
       * TrilinosWrappers::PreconditionAMG class, which is compatible also with
       * the deal.II native matrices that we are using here.
       *
       * The behaviour of the solve function is controlled by the `Solver`
       * section of the parameter file, i.e.,
       * @code{.sh}
       * subsection Solver
       *     set Absolute tolerance     = 1e-12
       *     set Consecutive iterations = 2
       *     set Log history            = false
       *     set Log result             = false
       *     set Maximum iterations     = 1000
       *     set Relative tolerance     = 1e-12
       *     set Solver control type    = tolerance
       *     set Solver name            = cg
       *     subsection AMG Preconditioner
       *       set Aggregation threshold = 0.0001
       *       set Coarse type           = Amesos-KLU
       *       set Elliptic              = true
       *       set High Order Elements   = false
       *       set Number of cycles      = 1
       *       set Output details        = false
       *       set Smoother overlap      = 0
       *       set Smoother sweeps       = 2
       *       set Smoother type         = Chebyshev
       *       set w-cycle               = false
       *     end
       *   end
       * end
       * @endcode
       */
      void
      solve();

      /**
       * Output the solution.
       *
       * When you have computed a solution, you probably want to do something
       * with it. For example, you may want to output it in a format that can be
       * visualized, or you may want to compute quantities you are interested
       * in: say, heat fluxes in a heat exchanger, air friction coefficients of
       * a wing, maximum bridge loads, or simply the value of the numerical
       * solution at a point. This function is therefore the place for
       * postprocessing your solution.
       *
       * We use the ParsedTools::DataOut class to postprocess our solution. The
       * behaviour of this function is controlled by the section `Output` of the
       * parameter file:
       * @code{.sh}
       * subsection Output
       *   set Curved cells region    = curved_inner_cells
       *   set Output format          = vtu
       *   set Output material ids    = true
       *   set Output partitioning    = true
       *   set Problem base name      = solution
       *   set Subdivisions           = 0
       *   set Write high order cells = true
       * end
       * @endcode
       *
       * This wrapper class is a bit more complicated than the one in step-3.
       * Although its usage is very similar, it is important to note that the
       * main difference between ParsedTools::DataOut and dealii::DataOut is
       * given by the fact that ParsedTools::DataOut is driven by the parameters
       * above, and simply calls the relevant functions of dealii::DataOut with
       * specific arguments, whereas dealii::DataOut is usually driven directly
       * by the source code, and requires you to recompile the code if you want,
       * for example, to change the output format, or to disable the writing of
       * high order cells, or to plot also the material ids or the partitioning
       * of the problem.
       */
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

      /**
       * \name Grid classes
       * @{
       */
      /**
       * A wrapper around dealii::GridIn, dealii::GridOut, and
       * dealii::GridGenerator namespace.
       *
       * The action of this class is driven by the section `Grid` of the
       * parameter file:
       * @code{.sh}
       * subsection Grid
       *   set Input name                = hyper_cube
       *   set Arguments                 = 0: 1: false
       *   set Initial grid refinement   = 0
       *   set Output name               =
       *   set Transform to simplex grid = false
       * end
       * @endcode
       *
       * Where you can specify what grid to generate, how to generate it, or
       * what file to read the grid from, and to what file to write the grid to,
       * in addition to the initial refinement of the grid.
       */
      ParsedTools::GridGenerator<dim, spacedim> grid_generator;

      /**
       * Grid refinement and error estimation.
       *
       * This class is a wrapper around the dealii::GridRefinement namespace,
       * and around the KellyErrorEstimator class. The action of this class is
       * governed by the section `Grid/Refinement` of the parameter file:
       * @code{.sh}
       * subsection Grid
       *   subsection Refinement
       *     set Number of refinement cycles = 1
       *     subsection Error estimator
       *       set Component mask =
       *       set Estimator type = kelly
       *     end
       *     subsection Marking strategy
       *       set Coarsening parameter                   = 0.1
       *       set Maximum level                          = 0
       *       set Maximum number of cells (if available) = 0
       *       set Minimum level                          = 0
       *       set Refinement parameter                   = 0.3
       *       set Refinement strategy                    = global
       *     end
       *   end
       * end
       * @endcode
       * where you can specify the number of refinement cycles, the type of
       * error estimator, what marking strategy to use, etc.
       *
       * At the moment, local refinement is only supported on quad/hex grids. If
       * you try to run the code with a local refinement strategy with a
       * tria/tetra grid, an exception will be thrown at run time.
       */
      ParsedTools::GridRefinement grid_refinement;

      /**
       * The actual triangulation.
       *
       * We allow also for the possibility to use a triangulation that is of
       * co-dimension one or two w.r.t. the space dimension. This is useful when
       * you want to solve Laplace-Beltrami problems, on surfaces, and by the
       * definition of the weak form that we use (which is using the tangent
       * gradient by default), this is what you would get if you use a
       * co-dimension one grid.
       */
      Triangulation<dim, spacedim> triangulation;
      /** @} */

      /**
       * \name Finite element and dof classes
       * @{
       */

      /**
       * A wrapper around deal.II dealii::FiniteElement classes.
       *
       * The action of this class is driven by the parameter `Finite element
       * space (u)` of the parameter file:
       * @code{.sh}
       * subsection Poisson
       *   set Finite element space (u) = FE_Q(1)
       * end
       * @endcode
       *
       * You should make sure that the type of finite element you specify
       * matches the type of triangulation you are using, i.e., FE_Q is
       * supported only on quad/hex grids, while FE_SimplexP is supported
       * only on tri/tetra grids.
       *
       * The syntax used to specify the finite element type is the same used by
       * the FETools::get_fe_by_name() function.
       */
      ParsedTools::FiniteElement<dim, spacedim> finite_element;

      /**
       * The actual DoFHandler class.
       */
      DoFHandler<dim, spacedim> dof_handler;

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
      AffineConstraints<double>    constraints;
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

      /**
       * Constants of the problem.
       *
       * Most of the problems we will work with, define constants of different
       * types (physical constants, material properties, numerical constants,
       * etc.). The  ParsedTools::Constants class allows you to define these in
       * a centralized way, and allows you to share these constants with all the
       * function definitions you may use in your code later on (e.g., the
       * forcing term, or the boundary conditions).
       *
       * The action of this class is driven by the section `Constants` of the
       * parameter file:
       * @code{.sh}
       * subsection Constants
       *   set Diffusion coefficient (kappa) = 1
       * end
       * @endcode
       * where you can specify the diffusion coefficient of the problem.
       *
       * Which constants are defined is built into the ParsedTools::Constants
       * class at construction time (see the source for the constructor
       * Poisson())
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

      /**
       * This is a wrapper around the dealii::ParsedConvergenceTable class, that
       * allows you to specify what error to computes, and how to compute them.
       *
       * The action of this class is driven by the section `Error table` of the
       * parameter file:
       * @code{.sh}
       * subsection Error table
       *   set Enable computation of the errors = true
       *   set Error file name                  =
       *   set Error precision                  = 3
       *   set Exponent for p-norms             = 2
       *   set Extra columns                    = cells, dofs
       *   set List of error norms to compute   = L2_norm, Linfty_norm, H1_norm
       *   set Rate key                         = dofs
       *   set Rate mode                        = reduction_rate_log2
       * end
       * @endcode
       *
       * The above code, for example, would produce a convergence table that
       * looks like
       * @code{.sh}
       * cells dofs    u_L2_norm    u_Linfty_norm    u_H1_norm
       *    16    25 1.190e-01    - 2.034e-01    - 1.997e+00    -
       *    64    81 3.018e-02 2.33 7.507e-02 1.70 1.003e+00 1.17
       *   256   289 7.587e-03 2.17 2.060e-02 2.03 5.031e-01 1.09
       *  1024  1089 1.900e-03 2.09 5.271e-03 2.06 2.518e-01 1.04
       *  4096  4225 4.751e-04 2.04 1.325e-03 2.04 1.259e-01 1.02
       * 16384 16641 1.188e-04 2.02 3.318e-04 2.02 6.296e-02 1.01
       * @endcode
       *
       * The table above can be used *as-is* to produce high quality pdf outputs
       * of your error convergence rates using the file in the FSI-suite
       * repository `latex/quick_convergence_graphs/graph.tex`. For example, the
       * above table would result in the following plot:
       *
       * @image html poisson_convergence_graph.png
       */
      ParsedTools::ConvergenceTable error_table;

      /**
       * Wrapper around the dealii::DataOut class.
       *
       * The action of this class is driven by the section `Output` of the
       * parameter file:
       * @code{.sh}
       * subsection Output
       *   set Curved cells region    = curved_inner_cells
       *   set Output format          = vtu
       *   set Output material ids    = true
       *   set Output partitioning    = true
       *   set Problem base name      = solution
       *   set Subdivisions           = 0
       *   set Write high order cells = true
       * end
       * @endcode
       *
       * A similar structure was used in the program dof_plotter.cc.
       *
       * For example, using the configuration specified above, a plot of the
       * solution using the `vtu` format would look like:
       *
       * @image html poisson_solution.png
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
      /** @} */
    };
  } // namespace Serial
} // namespace PDEs
#endif
