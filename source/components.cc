#include "parsed_tools/components.h"

#include <deal.II/base/patterns.h>

#include <numeric>
using namespace dealii;

namespace ParsedTools
{
  /**
   * Utilities for extracting components and boundary condition types from
   * strings.
   */
  namespace Components
  {
    unsigned int
    n_components(const std::string &component_names)
    {
      return Utilities::split_string_list(component_names).size();
    }



    std::string
    blocks_to_names(const std::vector<std::string> & components,
                    const std::vector<unsigned int> &multiplicities)
    {
      AssertDimension(components.size(), multiplicities.size());
      std::vector<std::string> all_components;
      for (unsigned int i = 0; i < components.size(); ++i)
        {
          for (unsigned int j = 0; j < multiplicities[i]; ++j)
            all_components.push_back(components[i]);
        }
      return Patterns::Tools::to_string(all_components);
    }



    std::pair<std::vector<std::string>, std::vector<unsigned int>>
    names_to_blocks(const std::string &component_names)
    {
      auto components = Utilities::split_string_list(component_names);
      auto blocks     = components;
      auto last       = std::unique(blocks.begin(), blocks.end());
      blocks.erase(last, blocks.end());
      std::vector<unsigned int> multiplicities;
      for (const auto c : blocks)
        multiplicities.push_back(
          std::count(components.begin(), components.end(), c));

      AssertDimension(blocks.size(), multiplicities.size());
      return std::make_pair(blocks, multiplicities);
    }

    std::string
    component_name(const std::string &component_names,
                   const std::string &selected_component)
    {
      const auto  components = Utilities::split_string_list(component_names);
      std::string name       = selected_component;
      // Try normal and tangential components
      {
        const auto c = Utilities::split_string_list(selected_component, ".");
        if (c.size() == 2)
          {
            if (c[1] == "n" || c[1] == "t" || c[1] == "N" || c[1] == "T")
              name = c[0];
            else
              {
                AssertThrow(false,
                            ExcMessage("You asked for " + selected_component +
                                       ", but I don' know how to interpret " +
                                       c[1]));
              }
          }
      }
      // Try direct name now
      if (std::find(components.begin(), components.end(), name) !=
          components.end())
        {
          return name;
        }
      // That didn't work. Se if we are asking for all components
      if (selected_component == "all" || selected_component == "ALL")
        {
          // We return the first component
          return components[0];
        }
      // Nothing else worked. Try numbers.
      else
        {
          try
            {
              unsigned int c = Utilities::string_to_int(selected_component);
              AssertThrow(c < components.size(),
                          ExcMessage(
                            "You asked for component " + selected_component +
                            ", but there are only " +
                            Utilities::int_to_string(components.size()) +
                            " components."));
              return components[c];
            }
          catch (...)
            {
              // Nothing else worked. Throw an exception.
              AssertThrow(false,
                          ExcMessage("You asked for " + selected_component +
                                     ", but I don' know how to interpret it " +
                                     "with names " + component_names));
            }
        }
      return "";
    }



    std::vector<unsigned int>
    block_indices(const std::string &component_names,
                  const std::string &selected_components)
    {
      const auto &[b, m] = names_to_blocks(component_names);
      const auto comps   = Utilities::split_string_list(selected_components);
      std::vector<unsigned int> indices;
      for (const auto &c : comps)
        {
          const auto name  = component_name(component_names, c);
          const auto index = std::find(b.begin(), b.end(), name);
          AssertThrow(index != b.end(),
                      ExcMessage("You asked for " + name +
                                 ", but I don' know how to interpret it " +
                                 "with names " + component_names));
          indices.push_back(std::distance(b.begin(), index));
        }
      return indices;
    }



    std::vector<unsigned int>
    component_indices(const std::string &component_names,
                      const std::string &selected_components)
    {
      const auto &[b, m] = names_to_blocks(component_names);
      const auto bi      = block_indices(component_names, selected_components);
      std::vector<unsigned int> indices;
      for (const auto &i : bi)
        indices.push_back(std::accumulate(m.begin(), m.begin() + i, 0));
      return indices;
    }



    std::pair<unsigned int, unsigned int>
    component_to_indices(const std::string &component_names,
                         const std::string &selected_component)
    {
      const auto i1 = component_indices(component_names, selected_component);
      const auto i2 = block_indices(component_names, selected_component);
      AssertDimension(i1.size(), i2.size());
      AssertThrow(i1.size() == 1,
                  ExcMessage("You asked for " + selected_component +
                             " components, but only one component at a time "
                             "should be passed to this function."));
      return std::make_pair(i1[0], i2[0]);
    }



    Type
    type(const std::string &component_names,
         const std::string &selected_component)
    {
      // Simple case: all components
      if (selected_component == "all" || selected_component == "ALL")
        return Type::all;

      const auto &[b, m] = names_to_blocks(component_names);
      {
        // Normal and tangential
        const auto c = Utilities::split_string_list(selected_component, ".");
        if (c.size() == 2)
          {
            if (c[1] == "n" || c[1] == "N")
              return Type::normal;
            else if (c[1] == "t" || c[1] == "T")
              return Type::tangential;
          }
      }
      // Now check if we have a scalar or a vector
      const auto &[ci, bi] =
        component_to_indices(component_names, selected_component);
      if (m[bi] == 1)
        return Type::scalar;
      else
        return Type::vector;
    }



    unsigned int
    n_blocks(const std::string &component_names)
    {
      const auto &[b, m] = names_to_blocks(component_names);
      return b.size();
    }



    ComponentMask
    mask(const std::string &component_names, const std::string &comp)
    {
      const unsigned int n = n_components(component_names);
      std::vector<bool>  _mask(n, false);
      // Treat differently the case of numbers and the case of strings
      try
        {
          const auto comps =
            Patterns::Tools::Convert<std::vector<unsigned int>>::to_value(comp);
          for (const auto &c : comps)
            {
              AssertThrow(c < n,
                          ExcMessage("You asked for component " +
                                     Utilities::int_to_string(c) +
                                     ", but there are only " +
                                     Utilities::int_to_string(n) +
                                     " components."));
              _mask[c] = true;
            }
          return ComponentMask(_mask);
        }
      catch (...)
        {
          // First the "ALL" case.
          if (comp == "ALL" || comp == "all")
            {
              for (unsigned int j = 0; j < n; ++j)
                _mask[j] = true;
              return ComponentMask(_mask);
            }
          // Then standard cases
          const auto ids     = component_indices(component_names, comp);
          const auto bids    = block_indices(component_names, comp);
          const auto &[b, m] = names_to_blocks(component_names);
          AssertDimension(ids.size(), bids.size());
          for (unsigned int i = 0; i < ids.size(); ++i)
            {
              _mask[ids[i]] = true;
              for (unsigned int j = 0; j < m[bids[i]]; ++j)
                _mask[ids[i] + j] = true;
            }
          return ComponentMask(_mask);
        }
    }
  } // namespace Components
} // namespace ParsedTools