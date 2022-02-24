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
     * This tutorial program is a revisitation of the step-8 tutorial program of
     * the deal.II library. Most of the documentation is copied from there.
     *
     * The major differences are related to the fact that we adapted the code to
     * the FSI-suite framework, and replaced many of deal.II default classes
     * with our wrappers in the ParsedTools namespace.
     *
     * One can write the elasticity equations in a number of ways. The one that
     * shows the symilarity with the Laplace equation in the most obvious way is
     * to write it as
     * @f[
     *   -
     *   \text{div}\, ({\mathbf C} \nabla \mathbf{u})
     *   =
     *   \mathbf f,
     * @f]
     *
     *
     * where $\mathbf u$ is a vector-valued displacement at each point, $\mathbf
     * f$ the force, and ${\mathbf C}$ is a rank-4 mixed tensor (i.e., it has
     * four indices, two of which are co-variant, and two are contra-variant)
     * that encodes the stress-strain relationship -- in essence, it represents
     * the <a href="https://en.wikipedia.org/wiki/Hooke%27s_law">"spring
     * constant"</a> in Hookes law that relates the displacement to the forces.
     * ${\mathbf C}$ will, in many cases, depend on $\mathbf x$ if the body
     * whose deformation we want to simulate is composed of different materials,
     * or a sinle material whose response is anysotropic (i.e., it is different
     * in different directions).
     *
     *
     * While the form of the equations above is correct, it is not the way they
     * are usually derived. In truth, the gradient of the displacement
     * $\nabla\mathbf u$ (a matrix) has no physical meaning whereas its
     * symmetrized version,
     * @f[ \varepsilon(\mathbf u)_{kl} =\frac{1}{2}(\partial_k u_l + \partial_l
     * u_k),
     * @f] does and is typically called the "infinitesimal strain". (Here and in
     * the following, $\partial_k=\frac{\partial}{\partial x_k}$. We will also
     * use the <a
     * href="https://en.wikipedia.org/wiki/Einstein_notation">Einstein summation
     * convention</a>, without distinguishing between lower and upper indices,
     * assuming a single global orthonormal basis for the Euclidean space). With
     * this definition of the strain, the elasticity equations then read as
     * @f[
     *   -
     *   \text{div}\, ({\mathbf C} \varepsilon(\mathbf u))
     *   =
     *   \mathbf f,
     * @f] which you can think of as the more natural generalization of the
     * Laplace equation to vector-valued problems. (The form shown first is
     * equivalent to this form because the tensor ${\mathbf C}$ has certain
     * symmetries, namely that $C_{ijkl}=C_{ijlk}$, and consequently ${\mathbf
     * C} \varepsilon(\mathbf u)_{kl} = {\mathbf C} \nabla\mathbf u$.)
     *
     * One can of course alternatively write these equations in component form:
     * @f[
     *   -
     *   \partial_j (C_{ijkl} \varepsilon_{kl})
     *   =
     *   f_i, \qquad i=1\ldots d.
     * @f]
     *
     * In many cases, one knows that the material under consideration is
     * isotropic, in which case by introduction of the two coefficients
     * $\lambda$ and $\mu$ the coefficient tensor reduces to
     * @f[ c_{ijkl}
     *   =
     *   \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik} \delta_{jl} +
     *   \delta_{il} \delta_{jk}).
     * @f]
     *
     * The elastic equations can then be rewritten in much simpler a form:
     * @f[
     *    -
     *    \nabla \lambda (\nabla\cdot {\mathbf u})
     *    -
     *    (\nabla \cdot \mu \nabla) {\mathbf u}
     *    -
     *    \nabla\cdot \mu (\nabla {\mathbf u})^T
     *    =
     *    {\mathbf f},
     * @f] and the respective bilinear form is then
     * @f[ a({\mathbf u}, {\mathbf v}) = \left( \lambda \nabla\cdot {\mathbf u},
     *   \nabla\cdot {\mathbf v} \right)_\Omega
     *   +
     *   \sum_{k,l} \left( \mu \partial_k u_l, \partial_k v_l \right)_\Omega
     *   +
     *   \sum_{k,l} \left( \mu \partial_k u_l, \partial_l v_k \right)_\Omega,
     * @f] or also writing the first term a sum over components:
     * @f[ a({\mathbf u}, {\mathbf v}) = \sum_{k,l} \left( \lambda \partial_l
     *   u_l, \partial_k v_k \right)_\Omega
     *   +
     *   \sum_{k,l} \left( \mu \partial_k u_l, \partial_k v_l \right)_\Omega
     *   +
     *   \sum_{k,l} \left( \mu \partial_k u_l, \partial_l v_k \right)_\Omega.
     * @f]
     *
     * @note As written, the equations above are generally considered to be the
     * right description for the displacement of three-dimensional objects if
     * the displacement is small and we can assume that <a
     * href="http://en.wikipedia.org/wiki/Hookes_law">Hooke's law</a> is valid.
     * In that case, the indices $i,j,k,l$ above all run over the set
     * $\{1,2,3\}$ (or, in the C++ source, over $\{0,1,2\}$). However, as is,
     * the program runs in 2d, and while the equations above also make
     * mathematical sense in that case, they would only describe a truly
     * two-dimensional solid. In particular, they are not the appropriate
     * description of an $x-y$ cross-section of a body infinite in the $z$
     * direction; this is in contrast to many other two-dimensional equations
     * that can be obtained by assuming that the body has infinite extent in
     * $z$-direction and that the solution function does not depend on the $z$
     * coordinate. On the other hand, there are equations for two-dimensional
     * models of elasticity; see for example the Wikipedia article on <a
     * href="http://en.wikipedia.org/wiki/Infinitesimal_strain_theory#Special_cases">plane
     * strain</a>, <a
     * href="http://en.wikipedia.org/wiki/Antiplane_shear">antiplane shear</a>
     * and <a href="http://en.wikipedia.org/wiki/Plane_stress#Plane_stress">plan
     * stress</a>.
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
