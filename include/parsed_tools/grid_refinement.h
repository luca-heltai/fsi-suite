#ifndef parsed_tools_grid_refinement_h
#define parsed_tools_grid_refinement_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/std_cxx20/iota_view.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/error_estimator.h>

#ifdef DEAL_II_WITH_MPI
#  ifdef DEAL_II_WITH_P4EST
#    include <deal.II/distributed/grid_refinement.h>
#    include <deal.II/distributed/tria.h>
#  endif
#endif

#include "parsed_tools/enum.h"

namespace ParsedTools
{
  /**
   * Refinement strategy implemented in the GridRefinement class.
   */
  enum class RefinementStrategy
  {
    global         = 1,
    fixed_fraction = 2,
    fixed_number   = 3,
  };

  /**
   * A wrapper for refinement strategies.
   *
   * This class implements a parametrized version of the ESTIMATE-MARK-REFINE
   * steps in AFEM methods. In particular, it allows you to select between
   *
   * - GridRefinement::refine_and_coarsen_fixed_fraction()
   * - GridRefinement::refine_and_coarsen_fixed_number()
   * - Triangulation::refine_global()
   *
   * and to use KellyErrorEstimator, or a custom estimator, to compute the error
   * indicators in the first two cases.
   *
   * In addition we have also some control parameters to guarantee that the
   * number of cells is maintained under a maximum, and that the refinement
   * level is kept between `min_level` and `max_level`.
   */
  class GridRefinement : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor.
     */
    GridRefinement(
      const std::string &       section_name        = "",
      const unsigned int &      n_refinement_cycles = 1,
      const RefinementStrategy &strategy         = RefinementStrategy::global,
      const std::string &       estimator_type   = "kelly",
      const double &            top_parameter    = .3,
      const double &            bottom_parameter = 0.1,
      const unsigned int &      max_cells        = 0,
      const int &               min_level        = 0,
      const int &               max_level        = 0,
      const std::map<std::string, std::function<void(dealii::Vector<float> &)>>
        &                          optional_estimators = {},
      const dealii::ComponentMask &component_mask = dealii::ComponentMask());

    /**
     * Mark cells a the triangulation for refinement or coarsening,
     * according to the given strategy applied to the supplied vector
     * representing local error criteria.
     *
     * Cells are only marked for refinement or coarsening. No refinement
     * is actually performed. You need to call
     * Triangulation::execute_coarsening_and_refinement() yourself.
     */
    template <int dim, int spacedim>
    void
    mark_cells(const dealii::Vector<float> &         criteria,
               dealii::Triangulation<dim, spacedim> &tria) const;

#ifdef DEAL_II_WITH_MPI
#  ifdef DEAL_II_WITH_P4EST
    /**
     * Mark cells of a distribtued triangulation for refinement or
     * coarsening, according to the given strategy applied to the
     * supplied vector representing local error criteria. If the
     * criterion which is specified in the parameter file is not
     * available, an exception is thrown.
     *
     * Cells are only marked for refinement or coarsening. No refinement
     * is actually performed. You need to call
     * Triangulation::execute_coarsening_and_refinement() yourself.
     */
    template <int dim, int spacedim>
    void
    mark_cells(
      const dealii::Vector<float> &                                criteria,
      dealii::parallel::distributed::Triangulation<dim, spacedim> &tria) const;
#  endif
#endif

    /**
     * Call the error estimator specified in the parameter file.
     */
    template <int dim, int spacedim, typename VectorType>
    void
    estimate_error(
      const dealii::Mapping<dim, spacedim> &   mapping,
      const dealii::DoFHandler<dim, spacedim> &dof_handler,
      const VectorType &                       solution,
      dealii::Vector<float> &                  estimated_error_per_cell,
      const std::map<
        dealii::types::boundary_id,
        const dealii::Function<spacedim, typename VectorType::value_type> *>
        &                               neumann_bc   = {},
      const dealii::Function<spacedim> *coefficients = nullptr,
      const typename dealii::KellyErrorEstimator<dim, spacedim>::Strategy
        strategy =
          dealii::KellyErrorEstimator<dim, spacedim>::cell_diameter_over_24)
      const;

    /**
     * Call the error estimator specified in the parameter file with a default
     * mapping.
     */
    template <int dim, int spacedim, typename VectorType>
    void
    estimate_error(
      const dealii::DoFHandler<dim, spacedim> &dof_handler,
      const VectorType &                       solution,
      dealii::Vector<float> &                  estimated_error_per_cell,
      const std::map<
        dealii::types::boundary_id,
        const dealii::Function<spacedim, typename VectorType::value_type> *>
        &                               neumann_bc   = {},
      const dealii::Function<spacedim> *coefficients = nullptr,
      const typename dealii::KellyErrorEstimator<dim, spacedim>::Strategy
        strategy =
          dealii::KellyErrorEstimator<dim, spacedim>::cell_diameter_over_24)
      const;

    /**
     * Get the current strategy object.
     */
    const RefinementStrategy &
    get_strategy() const
    {
      return strategy;
    }

    /**
     * Get the total number of refinemt cycles.
     */
    const unsigned int &
    get_n_refinement_cycles() const
    {
      return n_refinement_cycles;
    }

    /**
     * Get Return an object that can be thought of as an array containing all
     * indices from zero to @p n_refinement_cycles
     */
    dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>
    get_refinement_cycles() const
    {
      return dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>(
        0, n_refinement_cycles);
    }

    /**
     * Perform all the steps of the ESTIMATE-MARK-REFINE cycle.
     */
    template <int dim, int spacedim, typename VectorType, typename Tria>
    void
    estimate_mark_refine(const dealii::Mapping<dim, spacedim> &   mapping,
                         const dealii::DoFHandler<dim, spacedim> &dof_handler,
                         const VectorType &                       solution,
                         Tria &                                   tria) const;

    /**
     * Perform all the steps of the ESTIMATE-MARK-REFINE cycle.
     */
    template <int dim, int spacedim, typename VectorType, typename Tria>
    void
    estimate_mark_refine(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                         const VectorType &                       solution,
                         Tria &                                   tria) const;

  private:
    /**
     * Make sure that the refinement level is kept between `min_level` and
     * `max_level`.
     */
    template <int dim, int spacedim>
    void
    limit_levels(dealii::Triangulation<dim, spacedim> &tria) const;

    unsigned int       n_refinement_cycles;
    RefinementStrategy strategy;
    double             top_parameter;
    double             bottom_parameter;
    unsigned int       max_cells;
    int                min_level;
    int                max_level;

    std::string estimator_type;
    std::map<std::string, std::function<void(dealii::Vector<float> &)>>
                                optional_estimators;
    dealii::ComponentMask       component_mask;
    dealii::types::subdomain_id subdomain_id =
      dealii::numbers::invalid_material_id;
    dealii::types::material_id material_id =
      dealii::numbers::invalid_material_id;
  };

  // ================================================================
  // Template implementation
  // ================================================================
#ifndef DOXYGEN

#  ifdef DEAL_II_WITH_MPI
#    ifdef DEAL_II_WITH_P4EST
  template <int dim, int spacedim>
  void
  GridRefinement::mark_cells(
    const dealii::Vector<float> &                                criteria,
    dealii::parallel::distributed::Triangulation<dim, spacedim> &tria) const
  {
    if (strategy == RefinementStrategy::fixed_number)
      dealii::parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_number(
          tria,
          criteria,
          top_parameter,
          bottom_parameter,
          max_cells > 0 ? max_cells : std::numeric_limits<unsigned int>::max());
    else if (strategy == RefinementStrategy::fixed_fraction)
      dealii::parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_fraction(tria,
                                          criteria,
                                          top_parameter,
                                          bottom_parameter);
    else if (strategy == RefinementStrategy::global)
      for (const auto cell : tria.active_cell_iterators())
        cell->set_refine_flag();
    else
      Assert(false, dealii::ExcInternalError());
    limit_levels(tria);
  }
#    endif
#  endif



  template <int dim, int spacedim>
  void
  GridRefinement::mark_cells(const dealii::Vector<float> &         criteria,
                             dealii::Triangulation<dim, spacedim> &tria) const
  {
    if (strategy == RefinementStrategy::fixed_number)
      {
        if constexpr (dim == 1 && spacedim == 3)
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "Not instantiated for dim=1, spacedim=3"));
          }
        else
          {
            dealii::GridRefinement::refine_and_coarsen_fixed_number(
              tria,
              criteria,
              top_parameter,
              bottom_parameter,
              max_cells > 0 ? max_cells :
                              std::numeric_limits<unsigned int>::max());
          }
      }
    else if (strategy == RefinementStrategy::fixed_fraction)
      {
        if constexpr (dim == 1 && spacedim == 3)
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "Not instantiated for dim=1, spacedim=3"));
          }
        else
          {
            dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
              tria,
              criteria,
              top_parameter,
              bottom_parameter,
              max_cells > 0 ? max_cells :
                              std::numeric_limits<unsigned int>::max());
          }
      }
    else if (strategy == RefinementStrategy::global)
      {
        for (const auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_refine_flag();
      }
    else
      Assert(false, dealii::ExcInternalError());
    limit_levels(tria);
  }



  template <int dim, int spacedim>
  void
  GridRefinement::limit_levels(dealii::Triangulation<dim, spacedim> &tria) const
  {
    if (min_level != 0 || max_level != 0)
      for (const auto cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (min_level != 0 && cell->level() < min_level)
              cell->set_refine_flag();
            else if (max_level != 0 && cell->level() > max_level)
              cell->set_coarsen_flag();
            else if (max_level != 0 && cell->level() == max_level)
              cell->clear_refine_flag();
            else if (min_level != 0 && cell->level() == min_level)
              cell->clear_coarsen_flag();
          }
  }



  template <int dim, int spacedim, typename VectorType>
  void
  GridRefinement::estimate_error(
    const dealii::Mapping<dim, spacedim> &   mapping,
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                       solution,
    dealii::Vector<float> &                  estimated_error_per_cell,
    const std::map<
      dealii::types::boundary_id,
      const dealii::Function<spacedim, typename VectorType::value_type> *>
      &                               neumann_bc,
    const dealii::Function<spacedim> *coefficients,
    const typename dealii::KellyErrorEstimator<dim, spacedim>::Strategy
      strategy) const
  {
    if (this->strategy != RefinementStrategy::global)
      {
        AssertThrow(
          dof_handler.get_triangulation().all_reference_cells_are_hyper_cube(),
          dealii::ExcMessage(
            "Local refinement is only supported on quad-only or "
            "hex-only triangulations."));
        if (this->estimator_type == "kelly")
          {
            dealii::QGauss<dim - 1> face_quadrature_formula(
              dof_handler.get_fe().tensor_degree() + 1);

            estimated_error_per_cell.reinit(
              dof_handler.get_triangulation().n_active_cells());

            dealii::KellyErrorEstimator<dim, spacedim>::estimate(
              mapping,
              dof_handler,
              face_quadrature_formula,
              neumann_bc,
              solution,
              estimated_error_per_cell,
              this->component_mask,
              coefficients,
              dealii::numbers::invalid_unsigned_int,
              this->subdomain_id,
              this->material_id,
              strategy);
          }
        else
          {
            optional_estimators.at(this->estimator_type)(
              estimated_error_per_cell);
          }
      }
  }

  /**
   * Call the error estimator specified in the parameter file with a default
   * mapping.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  GridRefinement::estimate_error(
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                       solution,
    dealii::Vector<float> &                  estimated_error_per_cell,
    const std::map<
      dealii::types::boundary_id,
      const dealii::Function<spacedim, typename VectorType::value_type> *>
      &                               neumann_bc,
    const dealii::Function<spacedim> *coefficients,
    const typename dealii::KellyErrorEstimator<dim, spacedim>::Strategy
      strategy) const
  {
    const auto &mapping =
      dealii::get_default_linear_mapping(dof_handler.get_triangulation());
    estimate_error(mapping,
                   dof_handler,
                   solution,
                   estimated_error_per_cell,
                   neumann_bc,
                   coefficients,
                   strategy);
  }



  template <int dim, int spacedim, typename VectorType, typename Tria>
  void
  GridRefinement::estimate_mark_refine(
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                       solution,
    Tria &                                   tria) const
  {
    const auto &mapping =
      dealii::get_default_linear_mapping(dof_handler.get_triangulation());
    estimate_mark_refine(mapping, dof_handler, solution, tria);
  }



  template <int dim, int spacedim, typename VectorType, typename Tria>
  void
  GridRefinement::estimate_mark_refine(
    const dealii::Mapping<dim, spacedim> &   mapping,
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                       solution,
    Tria &                                   tria) const
  {
    // No estimates to do in global refinement case
    if (strategy == RefinementStrategy::global)
      tria.refine_global(1);
    else
      {
        dealii::Vector<float> estimated_error_per_cell;
        estimate_error(mapping,
                       dof_handler,
                       solution,
                       estimated_error_per_cell);
        mark_cells(estimated_error_per_cell, tria);
        tria.execute_coarsening_and_refinement();
      }
  }

#endif

} // namespace ParsedTools


#endif
