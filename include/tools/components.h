#ifndef components_h
#define components_h

#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include <deal.II/fe/component_mask.h>

namespace Tools
{
  namespace Components
  {
    /**
     * Count the number of components in the given list of comma separated
     * components.
     */
    unsigned int
    n_components(const std::string &component_names);

    /**
     * Count the number of components in the given list of comma separated
     * components.
     */
    unsigned int
    n_blocks(const std::string &component_names);


    /**
     * Build component names from block names and multiplicities.
     */
    std::string
    blocks_to_names(const std::vector<std::string> & components,
                    const std::vector<unsigned int> &multiplicities);

    /**
     * Build block names and multiplicities from component names.
     */
    std::pair<std::vector<std::string>, std::vector<unsigned int>>
    names_to_blocks(const std::string &component_names);

    /**
     * Return a component mask corresponding to a given selected component, from
     * a list of comma separated components.
     */
    dealii::ComponentMask
    mask(const std::string &component_names,
         const std::string &selected_component);
  } // namespace Components
} // namespace Tools
#endif