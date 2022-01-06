#ifndef components_h
#define components_h

#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include <deal.II/fe/component_mask.h>

#include "tools/parsed_enum.h"

namespace Tools
{
  namespace Components
  {
    /**
     * Enumerator used to identify patterns of components and their size in the
     * a block system.
     */
    enum class Type
    {
      all        = 0, //!< All components
      scalar     = 1, //!< Scalar
      vector     = 2, //!< Vector
      normal     = 3, //!< Normal component of a vector
      tangential = 4, //!< Tangential component of a vector
    };

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
     * Return the indices within the block corresponding to the given selected
     * components, or numbers::invalid_unsigned_int if the selected components
     * could not be found.
     *
     * Accepted patterns are comma separated versions of the following string
     * types:
     * - single component names, e.g., "p"
     * - block names, e.g., "u,p"
     * - component indices, e.g., "0,1"
     * - normal component names, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = block_indices(names, "u"); // {0}
     * r = block_indices(names, "u, p"); // {0,1}
     * r = block_indices(names, "p"); // {1}
     * r = block_indices(names, "2"); // {1}
     * r = block_indices(names, "1"); // {0}
     * r = block_indices(names, "0,1,2"); // {0,0,1}
     * r = block_indices(names, "u.n"); // {0}
     * r = block_indices(names, "u.t"); // {0}
     * @endcode
     */
    std::vector<unsigned int>
    block_indices(const std::string &component_names,
                  const std::string &selected_components);

    /**
     * Return the indices within the components corresponding to first component
     * of the given selected components, or numbers::invalid_unsigned_int if the
     * selected components could not be found.
     *
     * Accepted patterns are comma separated versions of the following string
     * types:
     * - single component names, e.g., "p"
     * - block names, e.g., "u,p"
     * - component indices, e.g., "0,1"
     * - normal component names, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_indices(names, "u"); // {0}
     * r = component_indices(names, "u, p"); // {0,2}
     * r = component_indices(names, "p"); // {2}
     * r = component_indices(names, "2"); // {2}
     * r = component_indices(names, "1"); // {0}
     * r = component_indices(names, "0,1,2"); // {0,0,1}
     * r = component_indices(names, "u.n"); // {0}
     * r = component_indices(names, "u.T"); // {0}
     * @endcode
     *
     * If the component names are unique, this is the same as the
     * block_indices() function, otherwise it returns the first component
     * indices.
     */
    std::vector<unsigned int>
    component_indices(const std::string &component_names,
                      const std::string &selected_components);

    /**
     * Return the indices within the components and within the blocks
     * corresponding to first component of the given selected component, or
     * numbers::invalid_unsigned_int if the selected components could not be
     * found.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the indices of the first component of the selected
     * component, and the index of the block, so, for "u" it would return {0,0}
     * and for "p" it would return {2,1} (third component, second block).
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_to_block_indices_map(names, "u"); // {0,0}
     * r = component_to_block_indices_map(names, "p"); // {2,1}
     * r = component_to_block_indices_map(names, "p"); // {2}
     * r = component_to_block_indices_map(names, "2"); // {2}
     * r = component_to_block_indices_map(names, "1"); // {0}
     * r = component_to_block_indices_map(names, "0,1,2"); // {0,0,1}
     * r = component_to_block_indices_map(names, "u.n"); // {0}
     * r = component_to_block_indices_map(names, "u.t"); // {0}
     * @endcode
     */
    std::pair<unsigned int, unsigned int>
    component_to_indices(const std::string &component_names,
                         const std::string &selected_component);

    /**
     * Return the canonical component name for the given selected component.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the canonical component name for an associated
     * pattern of components.
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_name(names, "u"); // "u"
     * r = component_name(names, "p"); // "p"
     * r = component_name(names, "1"); // "u"
     * r = component_name(names, "2"); // "p"
     * r = component_name(names, "u.n"); // "u"
     * @endcode
     */
    std::string
    component_name(const std::string &component_names,
                   const std::string &selected_component);

    /**
     * Return the component type for the given selected component.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the component type for an associated pattern of
     * components, according to the multiplicity of the given component, and to
     * the normal or tangential component.
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     * - a tangential component name, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_type(names, "u"); // vector
     * r = component_type(names, "p"); // scalar
     * r = component_type(names, "1"); // vector
     * r = component_type(names, "2"); // scalar
     * r = component_type(names, "u.n"); // normal
     * r = component_type(names, "u.t"); // tangential
     * @endcode
     */
    Type
    component_type(const std::string &component_name,
                   const std::string &selected_component);

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