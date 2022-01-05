#include "tools/parsed_boundary_conditions.h"

#include "tools/grid_info.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

namespace Tools
{
  template <int dim>
  ParsedBoundaryConditions<dim>::ParsedBoundaryConditions(
    const std::string &                          section_name,
    const std::string &                          component_names,
    const std::vector<IdsAndBoundaryConditions> &ids_and_bcs)
    : ParameterAcceptor(section_name)
    , component_names(component_names)
    , ids_and_bcs(std::move(ids_and_bcs))
    , grid_info(2)
  {
    std::string entry_name =
      "ids:components:bc_type:expressions (" + component_names + ")";
    add_parameter(entry_name, this->ids_and_bcs);
    enter_my_subsection(this->prm);
    this->prm.add_action(entry_name, [&](const std::string &) {
      functions.clear();
      for (const auto &id : ids_and_bcs)
        functions.push_back(
          std::make_unique<Functions::SymbolicFunction<dim>>(std::get<3>(id)));
      check_consistency();
    });
    leave_my_subsection(this->prm);
  }



  template <int dim>
  void
  ParsedBoundaryConditions<dim>::check_consistency() const
  {
    std::set<types::boundary_id> all_ids;
    unsigned int                 n_ids = 0;
    for (const auto &[ids, component, type, function] : ids_and_bcs)
      {
        all_ids.insert(ids.begin(), ids.end());
        n_ids += ids.size();
      }

    // Special meaning of boundary id numbers::invalid_boundary_id:
    if (all_ids.find(numbers::invalid_boundary_id) != all_ids.end())
      {
        AssertThrow(all_ids.size() == 1,
                    ExcMessage("If you use the boundary id -1, then no other "
                               "boundary ids can be specified"));
        AssertThrow(ids_and_bcs.size() == 1,
                    ExcMessage("Only one BoundaryCondition can be specified "
                               "with the special boundary id -1"));
      }
    else
      {
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
