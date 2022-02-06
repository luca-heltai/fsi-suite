#ifndef pdes_serial_fictitious_domain_h
#define pdes_serial_fictitious_domain_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "parsed_lac/amg.h"
#include "parsed_lac/inverse_operator.h"
#include "parsed_tools/boundary_conditions.h"
#include "parsed_tools/constants.h"
#include "parsed_tools/data_out.h"
#include "parsed_tools/finite_element.h"
#include "parsed_tools/function.h"
#include "parsed_tools/grid_generator.h"
#include "parsed_tools/grid_refinement.h"
#include "parsed_tools/mapping_eulerian.h"
using namespace dealii;
namespace PDEs
{
  namespace Serial
  {
    template <int dim, int spacedim = dim>
    class DistributedLagrange : public dealii::ParameterAcceptor
    {
    public:
      DistributedLagrange();
      void
      run();

    private:
      void
      generate_grids_and_fes();
      void
      adjust_embedded_grid(const bool apply_delta_refinement = true);
      void
      setup_dofs();
      void
      setup_coupling();
      void
      assemble_system();
      void
      solve();
      void
      output_results(const unsigned int cycle);

      // Const members go first
      const std::string component_names = "u";

      unsigned int coupling_quadrature_order = 3;
      unsigned int delta_refinement          = 0;
      unsigned int console_level             = 1;
      bool         use_direct_solver         = true;

      Triangulation<spacedim>                  space_grid;
      std::unique_ptr<FiniteElement<spacedim>> space_fe;
      std::unique_ptr<MappingFE<spacedim>>     space_mapping;
      std::unique_ptr<GridTools::Cache<spacedim, spacedim>>
                           space_grid_tools_cache;
      DoFHandler<spacedim> space_dh;

      Triangulation<dim, spacedim>                  embedded_grid;
      std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
      DoFHandler<dim, spacedim>                     embedded_dh;

      std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
      DoFHandler<dim, spacedim>                     embedded_configuration_dh;
      Vector<double>                                embedded_configuration;

      SparsityPattern           stiffness_sparsity;
      SparsityPattern           coupling_sparsity;
      SparseMatrix<double>      stiffness_matrix;
      SparseMatrix<double>      coupling_matrix;
      AffineConstraints<double> constraints;
      AffineConstraints<double> embedded_constraints;
      Vector<double>            solution;
      Vector<double>            rhs;
      Vector<double>            lambda;
      Vector<double>            embedded_rhs;
      Vector<double>            embedded_value;

      TimerOutput monitor;

      // Parameter members
      unsigned int finite_element_degree                        = 1;
      unsigned int embedded_space_finite_element_degree         = 1;
      unsigned int embedded_configuration_finite_element_degree = 1;

      // Then all parsed classes
      ParsedTools::GridGenerator<spacedim> grid_generator;
      ParsedTools::GridRefinement          grid_refinement;

      ParsedTools::GridGenerator<dim, spacedim>   embedded_grid_generator;
      ParsedTools::MappingEulerian<dim, spacedim> embedded_mapping;

      ParsedTools::Constants                    constants;
      ParsedTools::Function<spacedim>           embedded_value_function;
      ParsedTools::Function<spacedim>           forcing_term;
      ParsedTools::Function<spacedim>           exact_solution;
      ParsedTools::BoundaryConditions<spacedim> boundary_conditions;

      ParsedLAC::InverseOperator   stiffness_inverse_operator;
      ParsedLAC::AMGPreconditioner stiffness_preconditioner;

      ParsedLAC::InverseOperator schur_inverse_operator;

      mutable ParsedTools::DataOut<spacedim>      data_out;
      mutable ParsedTools::DataOut<dim, spacedim> embedded_data_out;
      ParsedConvergenceTable                      error_table;
    };
  } // namespace Serial

} // namespace PDEs

#endif