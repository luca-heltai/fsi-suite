#include "tools/components.h"

#include <deal.II/base/patterns.h>

using namespace dealii;

namespace Tools
{
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



    unsigned int
    n_blocks(const std::string &component_names)
    {
      const auto &[b, m] = names_to_blocks(component_names);
      return b.size();
    }



    ComponentMask
    mask(const std::string &component_names, const std::string &comp)
    {
      std::string selected_component;
      {
        // Try to extrac normal component first
        auto sc            = Utilities::split_string_list(comp, ".");
        selected_component = sc[0];
        if (sc.size() > 1)
          {
            AssertThrow(sc[1] == "N" || sc[1] == "n",
                        ExcMessage("Invalid normal component specification"));
            // Now check if the selected component is a vector
            const auto &[c, m] = names_to_blocks(component_names);
            const auto pos = std::find(c.begin(), c.end(), selected_component);
            AssertThrow(pos != c.end(),
                        ExcMessage("Invalid normal component specification"));
            AssertThrow(m[pos - c.begin()] > 1,
                        ExcMessage("Normal component specification is only "
                                   "valid for vector components"));
          }
      }
      const unsigned int n = n_components(component_names);
      std::vector<bool>  _mask(n, false);

      if (selected_component == "ALL" || selected_component == "all")
        {
          for (unsigned int j = 0; j < n; ++j)
            _mask[j] = true;
          return ComponentMask(_mask);
        }

      const auto components = Utilities::split_string_list(component_names);

      bool found = false;
      for (unsigned int c = 0; c < components.size(); ++c)
        {
          if (components[c] == selected_component)
            {
              _mask[c] = true;
              found    = true;
            }
        }
      if (found == true)
        return ComponentMask(_mask);

      // If we got here the selected component is not in the list of
      // components. Try to see if numbers were used.
      std::vector<unsigned int> selected_ids =
        Patterns::Tools::Convert<std::vector<unsigned int>>::to_value(
          selected_component);
      for (const auto i : selected_ids)
        {
          AssertThrow(i < n, ExcIndexRange(i, 0, n));
          _mask[i] = true;
          found    = true;
        }
      if (found == true)
        return ComponentMask(_mask);
      else
        AssertThrow(false,
                    ExcMessage("The selected component is not in the list "
                               "of components or the selected component "
                               "could not be converted to a list of "
                               "component ids."));
      return ComponentMask(); // Make compilers happy
    }
  } // namespace Components
} // namespace Tools