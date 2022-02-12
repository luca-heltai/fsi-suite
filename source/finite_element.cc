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



#include "parsed_tools/finite_element.h"

#include <deal.II/base/patterns.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_tools.templates.h>

#include <algorithm> // std::find

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  FiniteElement<dim, spacedim>::FiniteElement(
    const std::string &section_name,
    const std::string &_component_names,
    const std::string &fe_name)
    : ParameterAcceptor(section_name)
    , joint_component_names(_component_names)
    , component_names(Utilities::split_string_list(_component_names))
    , fe_name(fe_name)
  {
    component_blocks.resize(component_names.size());
    block_names.resize(component_names.size());
    unsigned int j = 0;
    for (unsigned int i = 0; i < component_names.size(); ++i)
      {
        if ((i > 0) && (component_names[i - 1] != component_names[i]))
          j++;
        component_blocks[i] = j;
        block_names[j]      = component_names[i];
      }
    block_names.resize(j + 1);

    add_parameter("Finite element space (" + joint_component_names + ")",
                  this->fe_name);
    enter_my_subsection(this->prm);
    this->prm.add_action(
      "Finite element space (" + joint_component_names + ")",
      [&](const std::string &value) {
        fe = FETools::get_fe_by_name<dim, spacedim>(value);
        // Check that the number of components is correct
        Assert(fe->n_components() == component_names.size(),
               ExcMessage("The number of components in the finite element "
                          "space does not match the number of components "
                          "for the FiniteElement."));
      });
    leave_my_subsection(this->prm);

    // Add some of the deal.II new FE classes.
    if (dim > 1)
      {
        FETools::add_fe_name<dim, spacedim>(
          "FE_SimplexP", new FETools::FEFactory<FE_SimplexP<dim, spacedim>>());
        FETools::add_fe_name<dim, spacedim>(
          "FE_SimplexDGP",
          new FETools::FEFactory<FE_SimplexDGP<dim, spacedim>>());
        FETools::add_fe_name<dim, spacedim>(
          "FE_SimplexP_Bubbles",
          new FETools::FEFactory<FE_SimplexP_Bubbles<dim, spacedim>>());
      }
  }



  template <int dim, int spacedim>
  FiniteElement<dim, spacedim>::operator dealii::FiniteElement<dim,
                                                               spacedim> &()
  {
    AssertThrow(fe, ExcNotInitialized());
    return *fe;
  }



  template <int dim, int spacedim>
  FiniteElement<dim, spacedim>::operator const dealii::
    FiniteElement<dim, spacedim> &() const
  {
    AssertThrow(fe, ExcNotInitialized());
    return *fe;
  }



  template <int dim, int spacedim>
  dealii::FiniteElement<dim, spacedim> &
  FiniteElement<dim, spacedim>::operator()()
  {
    AssertThrow(fe, ExcNotInitialized());
    return *fe;
  }



  template <int dim, int spacedim>
  const dealii::FiniteElement<dim, spacedim> &
  FiniteElement<dim, spacedim>::operator()() const
  {
    AssertThrow(fe, ExcNotInitialized());
    return *fe;
  }



  template <int dim, int spacedim>
  unsigned int
  FiniteElement<dim, spacedim>::n_components() const
  {
    return component_names.size();
  }


  template <int dim, int spacedim>
  unsigned int
  FiniteElement<dim, spacedim>::n_blocks() const
  {
    return block_names.size();
  }


  template <int dim, int spacedim>
  const std::string &
  FiniteElement<dim, spacedim>::get_joint_component_names() const
  {
    return joint_component_names;
  }


  template <int dim, int spacedim>
  const std::vector<std::string> &
  FiniteElement<dim, spacedim>::get_component_names() const
  {
    return component_names;
  }


  template <int dim, int spacedim>
  std::string
  FiniteElement<dim, spacedim>::get_block_names() const
  {
    return Patterns::Tools::to_string(block_names);
  }


  template <int dim, int spacedim>
  std::vector<unsigned int>
  FiniteElement<dim, spacedim>::get_component_blocks() const
  {
    return component_blocks;
  }

  template <int dim, int spacedim>
  unsigned int
  FiniteElement<dim, spacedim>::get_first_occurence(
    const std::string &var) const
  {
    auto pos_it =
      std::find(component_names.begin(), component_names.end(), var);
    Assert(pos_it != component_names.end(),
           ExcInternalError("Component not found!"));
    return pos_it - component_names.begin();
  }

  template <int dim, int spacedim>
  bool
  FiniteElement<dim, spacedim>::is_vector(const std::string &var) const
  {
    auto pos_it =
      std::find(component_names.begin(), component_names.end(), var);
    Assert(pos_it != component_names.end(),
           ExcInternalError("Component not found!"));
    pos_it++;
    if (pos_it == component_names.end())
      return false;
    return (*pos_it == var);
  }


  template class FiniteElement<1, 1>;
  template class FiniteElement<1, 2>;
  template class FiniteElement<1, 3>;
  template class FiniteElement<2, 2>;
  template class FiniteElement<2, 3>;
  template class FiniteElement<3, 3>;

} // namespace ParsedTools
