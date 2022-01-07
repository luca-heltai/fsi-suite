#include "tools/parsed_grid_refinement.h"

using namespace dealii;

namespace Tools
{
  ParsedGridRefinement::ParsedGridRefinement(
    const std::string &       section_name,
    const unsigned int &      n_refinement_cycles,
    const RefinementStrategy &strategy,
    const std::string &       estimator_type,
    const double &            top_parameter,
    const double &            bottom_parameter,
    const unsigned int &      max_cells,
    const int &               min_level,
    const int &               max_level,
    const std::map<std::string, std::function<void(dealii::Vector<float> &)>>
      &                          optional_estimators,
    const dealii::ComponentMask &component_mask)
    : ParameterAcceptor(section_name)
    , n_refinement_cycles(n_refinement_cycles)
    , strategy(strategy)
    , top_parameter(top_parameter)
    , bottom_parameter(bottom_parameter)
    , max_cells(max_cells)
    , min_level(min_level)
    , max_level(max_level)
    , estimator_type(estimator_type)
    , optional_estimators(optional_estimators)
    , component_mask(component_mask)
  {
    add_parameter("Number of refinement cycles", this->n_refinement_cycles);

    enter_subsection("Marking strategy");

    add_parameter("Refinement strategy", this->strategy);

    add_parameter("Refinement parameter",
                  this->top_parameter,
                  "Theta parameter, used to determine refinement fraction.",
                  this->prm,
                  Patterns::Double(0.0));

    add_parameter("Coarsening parameter",
                  this->bottom_parameter,
                  "Theta parameter, used to determine coearsening fraction.",
                  this->prm,
                  Patterns::Double(0.0));

    add_parameter("Maximum number of cells (if available)",
                  this->max_cells,
                  "Maximum number of cells.");

    add_parameter("Minimum level",
                  this->min_level,
                  "Any cell at refinement level below this number "
                  "will be marked for refinement.");

    add_parameter("Maximum level",
                  this->max_level,
                  "Any cell at refinement level above this number "
                  "will be marked for coarsening.");
    leave_subsection();

    enter_subsection("Error estimator");
    std::string selection = "kelly";
    for (const auto &[e, f] : optional_estimators)
      selection += "|" + e;
    add_parameter("Estimator type",
                  this->estimator_type,
                  "",
                  this->prm,
                  Patterns::Selection(selection));
    add_parameter("Component mask", this->component_mask);
    leave_subsection();
  }
} // namespace Tools
