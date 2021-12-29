

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

} // namespace Tools
