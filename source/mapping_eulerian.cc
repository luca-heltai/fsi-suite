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

#include "parsed_tools/mapping_eulerian.h"

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  MappingEulerian<dim, spacedim>::MappingEulerian(
    const DoFHandler<dim, spacedim> &dh,
    const std::string &              section_name,
    const std::string &              initial_configuration_or_displacement,
    const bool                       use_displacement,
    const ComponentMask &            mask)
    : ParameterAcceptor(section_name)
    , dof_handler(&dh)
    , mask(mask)
    , use_displacement(use_displacement)
    , initial_configuration_or_displacement_expression(
        initial_configuration_or_displacement)
  {
    add_parameter("Initial configuration or displacement",
                  this->initial_configuration_or_displacement_expression,
                  "The initial configuration of the mapping. If empty, the "
                  "identity configuration is used.");

    add_parameter(
      "Use displacement",
      this->use_displacement,
      "If true, the expression above is interpreted as a displacement, "
      "otherwise it is interpreted as a configuration.");
  }



  template <int dim, int spacedim>
  MappingEulerian<dim, spacedim>::operator const Mapping<dim, spacedim> &()
    const
  {
    AssertThrow(mapping,
                ExcMessage("You must call initialize() before using "
                           "the mapping."));
    return *mapping;
  }



  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  MappingEulerian<dim, spacedim>::operator()() const
  {
    AssertThrow(mapping,
                ExcMessage("You must call initialize() before using "
                           "the mapping."));
    return *mapping;
  }

  template class MappingEulerian<1, 1>;
  template class MappingEulerian<1, 2>;
  template class MappingEulerian<1, 3>;
  template class MappingEulerian<2, 2>;
  template class MappingEulerian<2, 3>;
  template class MappingEulerian<3, 3>;
} // namespace ParsedTools
