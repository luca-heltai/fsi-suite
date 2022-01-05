#ifndef parsed_boundary_conditions_h
#define parsed_boundary_conditions_h

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/symbolic_function.h>

#ifdef DEAL_II_WITH_SYMENGINE

#  include "tools/grid_info.h"
#  include "tools/parsed_enum.h"
#  include "tools/parsed_symbolic_function.h"

namespace Tools
{
  /**
   * Implemented boundary ids.
   */
  enum class BoundaryIdType
  {
    dirichlet = 0,
    // neumann           = 1,
    // robin             = 2,
    // dirichlet_nitsche = 3,
    // neumann_nitsche   = 4,
    // robin_nitsche     = 5,
    // normal_dirichlet  = 6,
  };

  /**
   * A wrapper for boundary conditions.
   *
   * This class can be used to store different types of boundary conditions,
   * applied to different ids of the domain boundary.
   *
   */
  template <int dim>
  class ParsedBoundaryConditions : public dealii::ParameterAcceptor
  {
  public:
    using IdsAndBoundaryConditions =
      std::tuple<std::set<dealii::types::boundary_id>,
                 std::string,
                 BoundaryIdType,
                 std::string>;

    /**
     * Constructor.
     */
    ParsedBoundaryConditions(
      const std::string &section_name    = "",
      const std::string &component_names = "u",
      const std::vector<std::set<dealii::types::boundary_id>> &ids =
        {{dealii::numbers::internal_face_boundary_id}},
      const std::vector<std::string> &   selected_components = {"u"},
      const std::vector<BoundaryIdType> &bc_type = {BoundaryIdType::dirichlet},
      const std::vector<std::string> &   expressions = {"0"});

    /**
     * Update the substitition map of every
     * dealii::Functions::SymbolicFunction defined in this object.
     *
     * See the documentation of
     * dealii::Functions::SymbolicFunction::update_user_substitution_map().
     */
    void
    update_substitution_map(
      const dealii::Differentiation::SD::types::substitution_map
        &substitution_map);

    /**
     * Call
     * dealii::Functions::SymbolicFunction::set_additional_function_arguments()
     * for every function defined in this object.
     *
     * See the documentation of
     * dealii::Functions::SymbolicFunction::set_additional_function_arguments().
     */
    void
    set_additional_function_arguments(
      const dealii::Differentiation::SD::types::substitution_map &arguments);

    /**
     * Update time in each dealii::Functions::SymbolicFunction defined in this
     * object.
     */
    void
    set_time(const double &time);

    /**
     * Check that the grid is compatible with this boundary condition object,
     * and that the boundary conditions are self consistent.
     */
    template <typename Tria>
    void
    check_consistency(const Tria &tria) const
    {
      grid_info.build_info(tria);
      check_consistency();
    }

    /**
     * Make sure the specified boundary conditions make sense. Do this,
     * independently of the Triangulation this object is associated with.
     */
    void
    check_consistency() const;

  private:
    /**
     * Component names of the boundary conditions.
     */
    const std::string component_names;

    /**
     * Number of components of the problem.
     */
    const unsigned int n_components;

    /**
     * Number of boundary conditions.
     */
    mutable unsigned int n_boundary_conditions;

    /**
     * Ids on which this object applies boundary conditions.
     */
    std::vector<std::set<dealii::types::boundary_id>> ids;

    /**
     * Component on which to apply the boundary condition.
     */
    std::vector<std::string> selected_components;

    /**
     * Type of boundary conditions.
     */
    std::vector<BoundaryIdType> bc_type;

    /**
     * Expressions for the boundary conditions.
     */
    std::vector<std::string> expressions;

    /**
     * The actual functions.
     */
    std::vector<std::unique_ptr<dealii::Functions::SymbolicFunction<dim>>>
      functions;

    /**
     * Information about the grid this BC applies to.
     */
    mutable GridInfo grid_info;
  };
} // namespace Tools
#endif
#endif