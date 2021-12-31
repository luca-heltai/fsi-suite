

#include "tools/parsed_constants.h"

using namespace dealii;

namespace Tools
{
  ParsedConstants::ParsedConstants(
    const std::string &             section_name,
    const std::vector<std::string> &names,
    const std::vector<double> &     default_values,
    const std::vector<std::string> &optional_names_for_parameter_file,
    const std::vector<std::string> &optional_documentation_strings)
    : ParameterAcceptor(section_name)
  {
    AssertDimension(names.size(), default_values.size());

    auto doc_strings = optional_documentation_strings;
    if (doc_strings.size() == 0)
      doc_strings.resize(names.size(), "");

    auto prm_names = optional_names_for_parameter_file;
    if (prm_names.size() == 0)
      prm_names = names;

    AssertDimension(names.size(), prm_names.size());

    AssertDimension(doc_strings.size(), default_values.size());

    // Before we do anything else, define standard math constants used in
    // dealii::numbers namespace.
    constants["E"]       = numbers::E;
    constants["LOG2E"]   = numbers::LOG2E;
    constants["LOG10E"]  = numbers::LOG10E;
    constants["LN2"]     = numbers::LN2;
    constants["LN10"]    = numbers::LN10;
    constants["PI"]      = numbers::PI;
    constants["PI_2"]    = numbers::PI_2;
    constants["PI_4"]    = numbers::PI_4;
    constants["SQRT2"]   = numbers::SQRT2;
    constants["SQRT1_2"] = numbers::SQRT1_2;

    for (unsigned int i = 0; i < names.size(); ++i)
      {
        constants[names[i]]                   = default_values[i];
        constants_parameter_entries[names[i]] = prm_names[i];
        constants_documentation[names[i]]     = doc_strings[i];

        auto entry = prm_names[i] == names[i] ?
                       prm_names[i] :
                       prm_names[i] + " (" + names[i] + ")";

        add_parameter(entry, constants[names[i]], doc_strings[i]);
      }
  }



  ParsedConstants::operator const std::map<std::string, double> &() const
  {
    return constants;
  }



  const double &
  ParsedConstants::operator[](const std::string &key) const
  {
    return constants.at(key);
  }



#ifdef DEAL_II_WITH_SYMENGINE
  /**
   * Return the constants defined in this class as a
   * Differentiation::SD::types::substitution_map.
   */
  ParsedConstants::operator const Differentiation::SD::types::
    substitution_map &() const
  {
    constant_substitution_map.clear();
    for (const auto &p : constants)
      constant_substitution_map[Differentiation::SD::Expression(p.first,
                                                                true)] =
        Differentiation::SD::Expression(p.second);
    return constant_substitution_map;
  }
#endif



} // namespace Tools
