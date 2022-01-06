
#ifndef poisson_include_file
#define poisson_include_file

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

#include "tools/parsed_boundary_conditions.h"
#include "tools/parsed_constants.h"
#include "tools/parsed_finite_element.h"
#include "tools/parsed_grid_generator.h"
#include "tools/parsed_grid_refinement.h"
#include "tools/parsed_inverse_operator.h"
#include "tools/parsed_symbolic_function.h"

/**
 * @page PDEs Collection of PDEs
 *
 * This namespace contains a collection of PDEs. All pdes are derived from
 * ParameterAcceptor, and use as many as possible of the objects defined in the
 * Tools namespace.
 */
namespace PDEs
{
  using namespace dealii;
  /**
   * Serial Poisson problem.
   *
   * Solve the Poisson equation in arbitrary dimensions and space dimensions.
   * When dim and spacedim are not the same, we solve the LaplaceBeltrami
   * equation.
   *
   * $$
   * \begin{cases}
   *  - \Delta u = f & \text{in} \Omega \subset R^{\text{spacedim}}\\
   * u = u_D & \text{on} \partial \Omega_D \\
   * \frac{\partial u}{\partial n} u = u_N & \text{on} \partial \Omega_N \\
   * \frac{\partial u}{\partial n} u + \rho u= u_R & \text{on} \partial \Omega_R
   * $$
   *
   * @ingroup PDEs
   */
  template <int dim, int spacedim = dim>
  class SerialPoisson : public ParameterAcceptor
  {
  public:
    SerialPoisson();

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
    Tools::ParsedGridGenerator<dim, spacedim> grid_generator;
    Tools::ParsedGridRefinement               grid_refinement;
    Triangulation<dim, spacedim>              triangulation;

    // FE and dofs classes
    Tools::ParsedFiniteElement<dim, spacedim> finite_element;
    DoFHandler<dim, spacedim>                 dof_handler;

    // Linear algebra classes
    AffineConstraints<double>    constraints;
    SparsityPattern              sparsity_pattern;
    SparseMatrix<double>         system_matrix;
    Vector<double>               solution;
    Vector<double>               system_rhs;
    Tools::ParsedInverseOperator inverse_operator;

    // Forcing terms and boundary conditions
    Tools::ParsedConstants                    constants;
    Tools::ParsedSymbolicFunction<spacedim>   forcing_term;
    Tools::ParsedSymbolicFunction<spacedim>   exact_solution;
    Tools::ParsedBoundaryConditions<spacedim> boundary_conditions;

    // Error convergence tables
    ParsedConvergenceTable error_table;

    unsigned int n_refinement_cycles = 1;
    std::string  output_filename     = "poisson";
  };
} // namespace PDEs
#endif
