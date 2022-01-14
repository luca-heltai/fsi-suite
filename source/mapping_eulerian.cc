#include "parsed_tools/mapping_eulerian.h"

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  MappingEulerian<dim, spacedim>::MappingEulerian(
    const std::string &              section_name,
    const DoFHandler<dim, spacedim> &dh,
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
    AssertThrow(mask == ComponentMask() ||
                  spacedim ==
                    mask.n_selected_components(dh.get_fe().n_components()),
                ExcMessage("The number of selected components in the "
                           "mask  must be " +
                           std::to_string(spacedim)));
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
