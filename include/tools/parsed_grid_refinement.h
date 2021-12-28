#ifndef parsed_grid_refinement_h
#define parsed_grid_refinement_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_MPI
#  ifdef DEAL_II_WITH_P4EST
#    include <deal.II/distributed/grid_refinement.h>
#    include <deal.II/distributed/tria.h>
#  endif
#endif


namespace Tools
{
  /**
   * A wrapper for refinement strategies.
   *
   * This class implements a parametrized version of the MARK step in AFEM
   * methods. In particular, it allows you to select between
   *
   * - GridRefinement::refine_and_coarsen_fixed_fraction()
   * - GridRefinement::refine_and_coarsen_fixed_number()
   * - Triangulation::refine_global()
   *
   * and adds also some control parameters to guarantee that the number of cells
   * is maintained under a maximum, and that the refinement level is kept
   * between `min_level` and `max_level`.
   */
  class ParsedGridRefinement : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor.
     */
    ParsedGridRefinement(const std::string & section_name  = "",
                         const std::string & strategy      = "fixed_fraction",
                         const double &      top_parameter = .3,
                         const double &      bottom_parameter = .1,
                         const unsigned int &max_cells        = 0,
                         const int &         min_level        = 0,
                         const int &         max_level        = 0);

    /**
     * Mark cells a the triangulation for refinement or coarsening,
     * according to the given strategy applied to the supplied vector
     * representing local error criteria.
     *
     * Cells are only marked for refinement or coarsening. No refinement
     * is actually performed. You need to call
     * Triangulation::execute_coarsening_and_refinement() yourself.
     */
    template <int dim, class Vector, int spacedim>
    void
    mark_cells(const Vector &                        criteria,
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
    template <int dim, class VectorType, int spacedim>
    void
    mark_cells(
      const VectorType &                                           criteria,
      dealii::parallel::distributed::Triangulation<dim, spacedim> &tria) const;
#  endif
#endif

  private:
    /**
     * Make sure that the refinement level is kept between `min_level` and
     * `max_level`.
     */
    template <int dim, int spacedim>
    void
    limit_levels(dealii::Triangulation<dim, spacedim> &tria) const;

    /**
     * Default expression of this function."
     */
    std::string  strategy;
    double       top_parameter;
    double       bottom_parameter;
    unsigned int max_cells;
    int          min_level;
    int          max_level;
  };

  // ================================================================
  // Template implementation
  // ================================================================

#ifdef DEAL_II_WITH_MPI
#  ifdef DEAL_II_WITH_P4EST
  template <int dim, class Vector, int spacedim>
  void
  ParsedGridRefinement::mark_cells(
    const Vector &                                               criteria,
    dealii::parallel::distributed::Triangulation<dim, spacedim> &tria) const
  {
    if (strategy == "fixed_number")
      dealii::parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_number(
          tria,
          criteria,
          top_parameter,
          bottom_parameter,
          max_cells ? max_cells : std::numeric_limits<unsigned int>::max());
    else if (strategy == "fixed_fraction")
      dealii::parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_fraction(tria,
                                          criteria,
                                          top_parameter,
                                          bottom_parameter);
    else if (strategy == "global")
      for (const auto cell : tria.active_cell_iterators())
        cell->set_refine_flag();
    else
      Assert(false, dealii::ExcInternalError());
    limit_levels(tria);
  }
#  endif
#endif



  template <int dim, class Vector, int spacedim>
  void
  ParsedGridRefinement::mark_cells(
    const Vector &                        criteria,
    dealii::Triangulation<dim, spacedim> &tria) const
  {
    if (strategy == "fixed_number")
      dealii::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        criteria,
        top_parameter,
        bottom_parameter,
        max_cells ? max_cells : std::numeric_limits<unsigned int>::max());
    else if (strategy == "fixed_fraction")
      dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        tria,
        criteria,
        top_parameter,
        bottom_parameter,
        max_cells ? max_cells : std::numeric_limits<unsigned int>::max());
    else if (strategy == "global")
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
  ParsedGridRefinement::limit_levels(
    dealii::Triangulation<dim, spacedim> &tria) const
  {
    if (min_level != 0 || max_level != 0)
      {
        for (const auto cell : tria.active_cell_iterators())
          {
            if (cell->level() < min_level)
              cell->set_refine_flag();
            else if (cell->level() > max_level)
              cell->set_coarsen_flag();
            else if (cell->level() == max_level)
              cell->clear_refine_flag();
            else if (cell->level() == min_level)
              cell->clear_coarsen_flag();
          }
      }
  }
} // namespace Tools


#endif
