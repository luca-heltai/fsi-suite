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
     * \f[ \begin{cases}
     *  - \Delta u = f & \text{ in } \Omega \subset R^{\text{spacedim}}\\
     *    u = u_D & \text{ on } \partial \Omega_D \\
     *    \frac{\partial u}{\partial n} = u_N & \text{ on } \partial \Omega_N
     *    \\
     *    \end{cases} \f]
     *
     * We will solve this equation on any grid that can be generated using the
     * ParsedTools::GridGenerator class. An example usage of that was class was
     * given in the file mesh_handler.cc.
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
     * <h3> *Assembling* the matrix and right hand side vector </h3>
     *
     * Now we know what we need (namely: objects that hold the matrix and
     * vectors, as well as ways to compute $A_{ij},F_i$), and we can look at
     * what it takes to make that happen:
     *
     * - The object for $A$ is of type SparseMatrix while those for $U$ and $F$
     *   are of type Vector. We will see in the program below what classes are
     *   used to solve linear systems.
     * - We need a way to form the integrals. In the finite element method, this
     *   is most commonly done using quadrature, i.e. the integrals are replaced
     *   by a weighted sum over a set of *quadrature points* on each cell. That
     *   is, we first split the integral over $\Omega$ into integrals over all
     *   cells,
     *   @f{align*} A_{ij} &= (\nabla\varphi_i, \nabla \varphi_j) = \sum_{K \in
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
     *     \sum_q \varphi_i(\mathbf x^\Gamma_q) f(\mathbf x^\Gamma_q)w^\Gamma_q,
     *   @f}
     *   where $\mathbb{T} \approx \Omega$ is a Triangulation approximating
     *   the domain, $\mathbf x^K_q$ is the $q$th quadrature point on cell $K$,
     *   and $w^K_q$ the $q$th quadrature weight. There are different parts to
     *   what is needed in doing this, and we will discuss them in turn next.
     * - First, we need a way to describe the location $\mathbf x_q^K$ of
     *   quadrature points and their weights $w^K_q$. They are usually mapped
     *   from the reference cell in the same way as shape functions, i.e.,
     *   implicitly using the MappingQ1 class or, if you explicitly say so,
     *   through one of the other classes derived from Mapping. The locations
     *   and weights on the reference cell are described by objects derived from
     *   the Quadrature base class. Typically, one chooses a quadrature formula
     *   (i.e. a set of points and weights) so that the quadrature exactly
     *   equals the integral in the matrix; this can be achieved because all
     *   factors in the integral are polynomial, and is done by Gaussian
     *   quadrature formulas, implemented in the QGauss class.
     * - We then need something that can help us evaluate $\varphi_i(\mathbf
     *   x^K_q)$ on cell $K$. This is what the FEValues class does: it takes a
     *   finite element objects to describe $\varphi$ on the reference cell, a
     *   quadrature object to describe the quadrature points and weights, and a
     *   mapping object (or implicitly takes the MappingQ1 class) and provides
     *   values and derivatives of the shape functions on the real cell $K$ as
     *   well as all sorts of other information needed for integration, at the
     *   quadrature points located on $K$. The process of computing the matrix
     *   and right hand side as a sum over all cells (and then a sum over
     *   quadrature points) is usually called *assembling the linear system*, or
     *   *assembly* for short, using the meaning of the word related to
     *   [assembly line](https://en.wikipedia.org/wiki/Assembly_line), meaning
     *   ["the act of putting together a set of pieces, fragments, or
     *   elements"](https://en.wiktionary.org/wiki/assembly).
     *
     * FEValues really is the central class in the assembly process. One way you
     * can view it is as follows: The FiniteElement and derived classes describe
     * shape <i>functions</i>, i.e., infinite dimensional objects: functions
     * have values at every point. We need this for theoretical reasons because
     * we want to perform our analysis with integrals over functions. However,
     * for a computer, this is a very difficult concept, since they can in
     * general only deal with a finite amount of information, and so we replace
     * integrals by sums over quadrature points that we obtain by mapping (the
     * Mapping object) using  points defined on a reference cell (the Quadrature
     * object) onto points on the real cell. In essence, we reduce the problem
     * to one where we only need a finite amount of information, namely shape
     * function values and derivatives, quadrature weights, normal vectors, etc,
     * exclusively at a finite set of points. The FEValues class is the one that
     * brings the three components together and provides this finite set of
     * information on a particular cell $K$. You will see it in action when we
     * assemble the linear system below.
     *
     * It is noteworthy that all of this could also be achieved if you simply
     * created these three objects yourself in an application program, and
     * juggled the information yourself. However, this would neither be simpler
     * (the FEValues class provides exactly the kind of information you actually
     * need) nor faster: the FEValues class is highly optimized to only compute
     * on each cell the particular information you need; if anything can be
     * re-used from the previous cell, then it will do so, and there is a lot of
     * code in that class to make sure things are cached wherever this is
     * advantageous.
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
       */
      Poisson();

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
      const std::string component_names = "u";

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
    };
  } // namespace Serial
} // namespace PDEs
#endif
