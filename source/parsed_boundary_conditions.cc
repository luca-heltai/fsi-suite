#include "tools/parsed_boundary_conditions.h"

#include "tools/components.h"
#include "tools/grid_info.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

namespace Tools
{
  template <int dim>
  ParsedBoundaryConditions<dim>::ParsedBoundaryConditions(
    const std::string &                                      section_name,
    const std::string &                                      component_names,
    const std::vector<std::set<dealii::types::boundary_id>> &ids,
    const std::vector<std::string> &   selected_components,
    const std::vector<BoundaryIdType> &bc_type,
    const std::vector<std::string> &   expressions)
    : ParameterAcceptor(section_name)
    , component_names(component_names)
    , n_components(Components::n_components(component_names))
    , ids(ids)
    , selected_components(selected_components)
    , bc_type(bc_type)
    , expressions(expressions)
    , grid_info(2)
  {
    Patterns::List expr_pattern(
      Patterns::List(Patterns::Anything(), 0, n_components, ";"),
      0,
      Patterns::List::max_int_value,
      "%");

    Patterns::List comp_pattern(
      Patterns::List(Patterns::Anything(), 0, n_components, ","),
      0,
      Patterns::List::max_int_value,
      ";");

    add_parameter("Boundary id sets (" + component_names + ")", this->ids);

    add_parameter("Selected components (" + component_names + ")",
                  this->selected_components,
                  "",
                  this->prm,
                  comp_pattern);

    add_parameter("Boundary condition types (" + component_names + ")",
                  this->bc_type);

    add_parameter("Expressions (" + component_names + ")",
                  this->expressions,
                  "",
                  this->prm,
                  expr_pattern);

    this->parse_parameters_call_back.connect([&]() {
      functions.clear();
      for (const auto &exp : this->expressions)
        functions.emplace_back(
          std::make_unique<dealii::Functions::SymbolicFunction<dim>>(exp));
      check_consistency();
    });
  }



  template <int dim>
  void
  ParsedBoundaryConditions<dim>::check_consistency() const
  {
    n_boundary_conditions = ids.size();
    AssertThrow(n_boundary_conditions == bc_type.size(),
                ExcMessage("The number of boundary ids must be equal to the "
                           "number of boundary condition types."));
    AssertThrow(n_boundary_conditions == expressions.size(),
                ExcMessage("The number of boundary ids must be equal to the "
                           "number of expressions."));
    AssertThrow(n_boundary_conditions == selected_components.size(),
                ExcMessage("The number of boundary ids "
                           "must be equal to the number "
                           "of selected components."));

    std::set<types::boundary_id> all_ids;
    unsigned int                 n_ids = 0;
    for (const auto &id : ids)
      {
        all_ids.insert(id.begin(), id.end());
        n_ids += id.size();
      }

    // Special meaning of boundary id numbers::invalid_boundary_id:
    if (all_ids.find(numbers::invalid_boundary_id) != all_ids.end())
      {
        AssertThrow(all_ids.size() == 1,
                    ExcMessage("If you use the boundary id -1, then no other "
                               "boundary ids can be specified"));
        AssertThrow(n_boundary_conditions == 1,
                    ExcMessage("Only one BoundaryCondition can be specified "
                               "with the special boundary id -1"));
      }
    else
      {
        AssertThrow(all_ids.size() == n_ids,
                    ExcMessage("You specified the same "
                               " boundary id more than once "
                               "in two different boundary conditions"));
        // We check consistency with the triangulation
        if (grid_info.n_active_cells > 0)
          {
            AssertThrow(all_ids.size() == grid_info.boundary_ids.size(),
                        ExcMessage("The number of boundary ids specified in "
                                   "the input file does not match the number "
                                   "of boundary ids in the triangulation"));
          }
      }
  }



  template <int dim>
  void
  ParsedBoundaryConditions<dim>::update_substitution_map(
    const dealii::Differentiation::SD::types::substitution_map
      &substitution_map)
  {
    for (auto &f : functions)
      f->update_user_substitution_map(substitution_map);
  }



  template <int dim>
  void
  ParsedBoundaryConditions<dim>::set_additional_function_arguments(
    const Differentiation::SD::types::substitution_map &arguments)
  {
    for (auto &f : functions)
      f->set_additional_function_arguments(arguments);
  }



  template <int dim>
  void
  ParsedBoundaryConditions<dim>::set_time(const double &time)
  {
    for (auto &f : functions)
      f->set_time(time);
  }



  template class ParsedBoundaryConditions<1>;
  template class ParsedBoundaryConditions<2>;
  template class ParsedBoundaryConditions<3>;
} // namespace Tools
#endif
