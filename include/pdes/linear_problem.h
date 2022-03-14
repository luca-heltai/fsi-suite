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

// Make sure we don't redefine things
#ifndef base_serial_linear_problem_include_file
#define base_serial_linear_problem_include_file

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/signals2.hpp>

#include <fstream>
#include <iostream>

#include "lac.h"

#define FORCE_USE_OF_TRILINOS

namespace LA
{
  using namespace dealii::LinearAlgebraDealII;
} // namespace LA

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
  /**
   * Construct a LinearProblem.
   */
  template <int dim, int spacedim = dim, class LacType = LAC::LAdealii>
  class LinearProblem : public ParameterAcceptor
  {
  public:
    /**
     * Constructor. Store component names and component masks.
     */
    LinearProblem(const std::string &component_names = "u",
                  const std::string &problem_name    = "");

    /**
     * Virtual destructor.
     */
    virtual ~LinearProblem() = default;

    /**
     * Main entry point of the problem.
     */
    virtual void
    run();

    /**
     * Default CopyData object, used in the WorkStream class.
     */
    using CopyData = MeshWorker::CopyData<1, 1, 1>;

    /**
     * Default ScratchData object, used in the workstream class.
     */
    using ScratchData = MeshWorker::ScratchData<dim, spacedim>;

    /**
     * Block vector type.
     */
    using BlockVectorType = typename LacType::BlockVector;

    /**
     * Vector type.
     */
    using VectorType = typename BlockVectorType::BlockType;

    /**
     * Block matrix type.
     */
    using BlockMatrixType = typename LacType::BlockSparseMatrix;

  protected:
    /**
     * Assemble the local system matrix on `cell`, using `scratch` for
     * FEValues and other expensive scratch objects, and store the result in
     * the `copy` object. See the documentation of WorkStream for an
     * explanation of how to use this function.
     *
     * @param cell Cell on which we assemble the local matrix and rhs.
     * @param scratch Scratch object.
     * @param copy Copy object.
     */
    virtual void
    assemble_system_one_cell(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      ScratchData &                                                   scratch,
      CopyData &                                                      copy);

    /**
     * Distribute the data that has been assembled by
     * assemble_system_on_cell() to the global matrix and rhs.
     *
     * @param copy The local data to distribute on the system matrix and rhs.
     */
    virtual void
    copy_one_cell(const CopyData &copy);

    /**
     * Solve the global system.
     */
    virtual void
    solve();

    /**
     * Perform a posteriori error estimation, and store the results in the
     * `error_per_cell` vector.
     */
    virtual void
    estimate(Vector<float> &error_per_cell) const;

    /**
     * According to the chosen strategy, mark some cells for refinement,
     * possibily using the `error_per_cell` vector.
     */
    void
    mark(const Vector<float> &error_per_cell);

    /**
     * Refine the grid.
     */
    void
    refine();

    /**
     * Initial setup: distribute degrees of freedom, make all vectors and
     * matrices of the right size, initialize functions and pointers.
     */
    virtual void
    setup_system();

    /**
     * A signal that is called at the end of setup_system()
     */
    boost::signals2::signal<void()> add_constraints_call_back;

    /**
     * A signal that is called at the end of setup_system()
     */
    boost::signals2::signal<void()> setup_system_call_back;

    /**
     * Actually loop over cells, and assemble the global system.
     */
    virtual void
    assemble_system();

    /**
     * A signal that is called at the end of assemble_system()
     */
    boost::signals2::signal<void()> assemble_system_call_back;

    /**
     * Output the solution and the grid in a format that can be read by
     * Paraview or Visit.
     *
     * @param cycle A number identifying the refinement cycle.
     */
    virtual void
    output_results(const unsigned cycle) const;

    /**
     * print some information about the current processes/mpi/ranks/etc.
     */
    virtual void
    print_system_info() const;

    /**
     * Connect to this signal to add data additional vectors to the output
     * system.
     */
    boost::signals2::signal<void(ParsedTools::DataOut<dim, spacedim> &)>
      add_data_vector;

    /**
     * Comma seperated names of components.
     */
    const std::string component_names;

    /**
     * Number of components.
     */
    const unsigned int n_components;

    /**
     * Name of the problem to solve.
     */
    const std::string problem_name;

    /**
     * Name of the section to use within the parameter file.
     */
    const std::string section_name;

    /**
     * Global mpi communicator.
     */
    MPI_Comm mpi_communicator;

    /**
     * Number of threads to use for multi-threaded assembly.
     */
    int number_of_threads = -1;

    /**
     * Output only on processor zero.
     */
    ConditionalOStream pcout;

    /**
     * Timing information.
     */
    mutable TimerOutput timer;

    /**
     * A wrapper around GridIn, GridOut, and
     * GridGenerator namespace.
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
     * what file to read the grid from, and to what file to write the grid
     * to, in addition to the initial refinement of the grid.
     */
    ParsedTools::GridGenerator<dim, spacedim> grid_generator;

    /**
     * Grid refinement and error estimation.
     *
     * This class is a wrapper around the GridRefinement namespace,
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
     * At the moment, local refinement is only supported on quad/hex grids.
     * If you try to run the code with a local refinement strategy with a
     * tria/tetra grid, an exception will be thrown at run time.
     */
    ParsedTools::GridRefinement grid_refinement;

    /**
     * The problem triangulation.
     */
    parallel::distributed::Triangulation<dim, spacedim> triangulation;


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
     * The syntax used to specify the finite element type is the same used
     * by the FETools::get_fe_by_name() function.
     */
    ParsedTools::FiniteElement<dim, spacedim> finite_element;

    /**
     * The Mapping between reference and real elements.
     *
     * This is a unique pointer to allow creation via parameter files.
     */
    std::unique_ptr<Mapping<dim, spacedim>> mapping;

    /**
     * Handler of degrees of freedom.
     */
    DoFHandler<dim, spacedim> dof_handler;

    /**
     * Hanging nodes and essential boundary conditions.
     */
    AffineConstraints<double> constraints;

    /**
     * Dofs per block
     */
    std::vector<types::global_dof_index> dofs_per_block;

    /**
     * All degrees of freedom owned by this MPI process.
     */
    std::vector<IndexSet> locally_owned_dofs;

    /**
     * All degrees of freedom needed for output and error estimation.
     */
    std::vector<IndexSet> locally_relevant_dofs;

    /**
     * System sparsity pattern.
     */
    typename LacType::BlockSparsityPattern system_block_sparsity;

    /**
     * System matrix.
     */
    typename LacType::BlockSparseMatrix system_block_matrix;

    /**
     * A read only copy of the solution vector used for output and error
     * estimation.
     */
    typename LacType::BlockVector locally_relevant_block_solution;

    /**
     * Solution vector.
     */
    typename LacType::BlockVector block_solution;

    /**
     * The system right hand side. Read-write vector, containing only
     * locally owned dofs.
     */
    typename LacType::BlockVector system_block_rhs;

    /**
     * Storage for local error estimator. This vector contains also values
     * associated to  artificial cells (i.e., it is of length
     * `triangulation.n_active_cells()`), but it is non-zero only on locally
     * owned cells. The estimate() method only fills locally owned cells.
     */
    Vector<float> error_per_cell;

    /**
     * Inverse operator.
     */
    ParsedLAC::InverseOperator inverse_operator;

    /**
     * Preconditioner.
     */
    ParsedLAC::AMGPreconditioner preconditioner;

    /**
     * The actual function to use as a forcing term. This is a wrapper
     * around the dealii::ParsedFunction class, which allows you to define a
     * function through a symbolic expression (a string) in a parameter
     * file.
     *
     * The action of this class is driven by the section `Functions`, with
     * the parameter `Forcing term`:
     * @code{.sh}
     * subsection Functions
     *  set Forcing term = kappa*8*PI^2*sin(2*PI*x)*sin(2*PI*y)
     * end
     * @endcode
     *
     * You can use any of the numerical constants that are defined in the
     * dealii::numbers namespace, such as PI, E, etc, as well as the
     * constants defined at construction time in the ParsedTools::Constants
     * class.
     */
    ParsedTools::Function<spacedim> forcing_term;

    /**
     * The actual function to use as a exact solution when computing the
     * errors. This is a wrapper around the dealii::ParsedFunction class,
     * which allows you to define a function through a symbolic expression
     * (a string) in a parameter file.
     *
     * The action of this class is driven by the section `Functions`, with
     * the parameter `Exact solution`:
     * @code{.sh}
     * subsection Functions
     *  set Exact solution = sin(2*PI*x)*sin(2*PI*y)
     * end
     * @endcode
     *
     * You can use any of the numerical constants that are defined in the
     * dealii::numbers namespace, such as PI, E, etc, as well as the
     * constants defined at construction time in the ParsedTools::Constants
     * class.
     */
    ParsedTools::Function<spacedim> exact_solution;

    /**
     * Boundary conditions used in this class.
     *
     * The action of this class is driven by the section `Boundary
     * conditions` of the parameter file:
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
     * following: for every set of boundary ids of the triangulation, you
     * need to specify what boundary conditions are assumed to be imposed on
     * that set. If you only want to specify one type of boundary condition
     * (`dirichlet` or `neumann`) on all of the boundary, you can do so by
     * specifying `-1` as the boundary id set.
     *
     * Multiple boundary conditions can be specified, but the same id should
     * should appear only once in the parameter file (i.e., you cannot apply
     * different types of boundary conditions on the same boundary id).
     *
     * Keep in mind the following caveats:
     * - Boundary conditions are specified as comma separated strings, so
     * you can specify "set Boundary condition types (u) = neumann,
     * dirichlet" for two different sets of boundary ids.
     * - Following the previous example, different boundary id sets are
     *   separated by a semicolumn, and in each set, different boundary ids
     *   are separated by a column, so, for example, if you specify as
     *   `set Boundary id sets (u) = 0, 1; 2, 3`, then boundary ids 0 and 1
     *   will get Neumann boundary conditions, while boundary ids 2 and 3
     * will get Dirichlet boundary conditions.
     * - Since an expression can contain a `,` character, then expression
     * for each component are separated by a semicolumn, and for each
     * boundary id set, are separated by the `%` character. For example, if
     * you want to specify homogeneous Neumann boundary conditions, and
     * constant Dirichlet boundary conditions you can set the following
     * parameter: `set Expressions (u) = 0 % 1`.
     * - The selected components, again can be `all`, a component name, or
     *   `u.n`, or `u.t` to select normal component, or tangential component
     *   in a vector valued problem. For scalar problems, only the name of
     * the component makes sense. This field allows you to control which
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
     * and 3, and homogeneous Neumann boundary conditions on the boundary
     * ids 0 and 1.
     */
    ParsedTools::BoundaryConditions<spacedim> boundary_conditions;


    /**
     * This is a wrapper around the dealii::ParsedConvergenceTable class,
     * that allows you to specify what error to computes, and how to compute
     * them.
     *
     * The action of this class is driven by the section `Error table` of
     * the parameter file:
     * @code{.sh}
     * subsection Error table
     *   set Enable computation of the errors = true
     *   set Error file name                  =
     *   set Error precision                  = 3
     *   set Exponent for p-norms             = 2
     *   set Extra columns                    = cells, dofs
     *   set List of error norms to compute   = L2_norm, Linfty_norm,
     * H1_norm set Rate key                         = dofs set Rate mode =
     * reduction_rate_log2 end
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
     * The table above can be used *as-is* to produce high quality pdf
     * outputs of your error convergence rates using the file in the
     * FSI-suite repository `latex/quick_convergence_graphs/graph.tex`. For
     * example, the above table would result in the following plot:
     *
     * @image html poisson_convergence_graph.png
     */
    mutable ParsedTools::ConvergenceTable error_table;

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
  };
} // namespace PDEs
#endif