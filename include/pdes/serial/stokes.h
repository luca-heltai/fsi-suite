#ifndef pdes_serial_stokes_h
#define pdes_serial_stokes_h

#include <deal.II/base/config.h>

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
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "parsed_lac/amg.h"
#include "parsed_lac/ilu.h"
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
     * Serial Stokes problem.
     *
     * @addtogroup cfd
     *
     * Solve the Stokes equation in arbitrary dimensions.
     *
     */
    template <int dim>
    class Stokes : public ParameterAcceptor
    {
    public:
      Stokes();

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
      ParsedTools::GridGenerator<dim> grid_generator;
      ParsedTools::GridRefinement     grid_refinement;
      Triangulation<dim>              triangulation;

      // FE and dofs classes
      ParsedTools::FiniteElement<dim> finite_element;
      DoFHandler<dim>                 dof_handler;
      std::unique_ptr<Mapping<dim>>   mapping;

      /**
       * Dofs per block
       */
      std::vector<types::global_dof_index> dofs_per_block;

      // Linear algebra classes
      AffineConstraints<double>    constraints;
      BlockSparsityPattern         sparsity_pattern;
      BlockSparseMatrix<double>    system_matrix;
      BlockVector<double>          solution;
      BlockVector<double>          system_rhs;
      ParsedLAC::InverseOperator   inverse_operator;
      ParsedLAC::AMGPreconditioner velocity_preconditioner;
      ParsedLAC::AMGPreconditioner schur_preconditioner;

      // Forcing terms and boundary conditions
      ParsedTools::Constants               constants;
      ParsedTools::Function<dim>           forcing_term;
      ParsedTools::Function<dim>           exact_solution;
      ParsedTools::BoundaryConditions<dim> boundary_conditions;

      // Error convergence tables
      ParsedConvergenceTable error_table;

      // Output class
      mutable ParsedTools::DataOut<dim> data_out;

      // Console level
      unsigned int console_level = 1;

      // Extractor for vector components
      const FEValuesExtractors::Vector velocity;
      const FEValuesExtractors::Scalar pressure;
    };
  } // namespace Serial
} // namespace PDEs
#endif