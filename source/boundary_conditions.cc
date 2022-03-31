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


#include "parsed_tools/boundary_conditions.h"

#include "parsed_tools/components.h"
#include "parsed_tools/grid_info.h"
#include "parsed_tools/patterns_unsigned_int.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

namespace ParsedTools
{
  template <int spacedim>
  BoundaryConditions<spacedim>::BoundaryConditions(
    const std::string &                                      section_name,
    const std::string &                                      component_names,
    const std::vector<std::set<dealii::types::boundary_id>> &ids,
    const std::vector<std::string> &          selected_components,
    const std::vector<BoundaryConditionType> &bc_type,
    const std::vector<std::string> &          expressions)
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

    Patterns::List id_pattern(Patterns::List(Patterns::UnsignedInteger(),
                                             0,
                                             Patterns::List::max_int_value,
                                             ","),
                              0,
                              Patterns::List::max_int_value,
                              ";");

    add_parameter("Boundary id sets (" + component_names + ")",
                  this->ids,
                  "",
                  this->prm,
                  id_pattern);

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
      // Parse expressions into functions.
      functions.clear();
      for (const auto &exp : this->expressions)
        functions.emplace_back(
          std::make_unique<dealii::Functions::SymbolicFunction<spacedim>>(exp));

      // Parse components into masks and types
      masks.clear();
      types.clear();
      for (const auto &comp : this->selected_components)
        {
          masks.push_back(Components::mask(this->component_names, comp));
          types.push_back(Components::type(this->component_names, comp));
        }

      // Check that everything is consistent
      check_consistency();
      update_user_substitution_map({});
    });
  }



  template <int spacedim>
  void
  BoundaryConditions<spacedim>::check_consistency() const
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
    // This is no longer the case for complex types: we could have a boundary id
    // specified for one component, and another one specified for a different
    // component.
    // else
    // {
    //   AssertThrow(all_ids.size() == n_ids,
    //               ExcMessage("You specified the same "
    //                          " boundary id more than once "
    //                          "in two different boundary conditions"));
    //   // We check consistency with the triangulation
    //   if (grid_info.n_active_cells > 0)
    //     {
    //       AssertThrow(all_ids.size() == grid_info.boundary_ids.size(),
    //                   ExcMessage("The number of boundary ids specified in "
    //                              "the input file does not match the number "
    //                              "of boundary ids in the triangulation"));
    //     }
    // }

    // Now check that the types are valid in this dimension
    AssertDimension(n_boundary_conditions, types.size());
    AssertDimension(n_boundary_conditions, masks.size());
    for (unsigned int i = 0; i < n_boundary_conditions; ++i)
      {
        if (types[i] == Components::Type::vector ||
            types[i] == Components::Type::normal ||
            types[i] == Components::Type::tangential)
          {
            AssertThrow(masks[i].n_selected_components() == spacedim,
                        ExcDimensionMismatch(masks[i].n_selected_components(),
                                             spacedim));
          }

        if (types[i] == Components::Type::normal ||
            types[i] == Components::Type::tangential)
          {
            AssertThrow(functions[i]->n_components == spacedim,
                        ExcDimensionMismatch(functions[i]->n_components,
                                             spacedim));
          }
        else
          {
            // In all other cases, it is the mask that determines the number
            // of of components, but the function needs to be of the correct
            // dimension
            AssertThrow(functions[i]->n_components == n_components,
                        ExcDimensionMismatch(functions[i]->n_components,
                                             n_components));
          }
      }
  }



  template <int spacedim>
  void
  BoundaryConditions<spacedim>::update_user_substitution_map(
    const dealii::Differentiation::SD::types::substitution_map
      &substitution_map)
  {
    auto smap       = substitution_map;
    smap["E"]       = numbers::E;
    smap["LOG2E"]   = numbers::LOG2E;
    smap["LOG10E"]  = numbers::LOG10E;
    smap["LN2"]     = numbers::LN2;
    smap["LN10"]    = numbers::LN10;
    smap["PI"]      = numbers::PI;
    smap["PI_2"]    = numbers::PI_2;
    smap["PI_4"]    = numbers::PI_4;
    smap["SQRT2"]   = numbers::SQRT2;
    smap["SQRT1_2"] = numbers::SQRT1_2;
    for (auto &f : functions)
      f->update_user_substitution_map(smap);
  }



  template <int spacedim>
  void
  BoundaryConditions<spacedim>::set_additional_function_arguments(
    const Differentiation::SD::types::substitution_map &arguments)
  {
    for (auto &f : functions)
      f->set_additional_function_arguments(arguments);
  }



  template <int spacedim>
  void
  BoundaryConditions<spacedim>::set_time(const double &time)
  {
    for (auto &f : functions)
      f->set_time(time);
  }



  template <int spacedim>
  std::set<dealii::types::boundary_id>
  BoundaryConditions<spacedim>::get_essential_boundary_ids() const
  {
    std::set<dealii::types::boundary_id> essential_boundary_ids;
    for (unsigned int i = 0; i < ids.size(); ++i)
      {
        if (bc_type[i] == BoundaryConditionType::dirichlet ||
            bc_type[i] == BoundaryConditionType::first_dof)
          essential_boundary_ids.insert(ids[i].begin(), ids[i].end());
      }
    return essential_boundary_ids;
  }



  template <int spacedim>
  std::set<dealii::types::boundary_id>
  BoundaryConditions<spacedim>::get_natural_boundary_ids() const
  {
    std::set<dealii::types::boundary_id> natural_boundary_ids;
    for (unsigned int i = 0; i < ids.size(); ++i)
      {
        if (bc_type[i] == BoundaryConditionType::neumann)
          natural_boundary_ids.insert(ids[i].begin(), ids[i].end());
      }
    return natural_boundary_ids;
  }



  template class BoundaryConditions<1>;
  template class BoundaryConditions<2>;
  template class BoundaryConditions<3>;
} // namespace ParsedTools
#endif
