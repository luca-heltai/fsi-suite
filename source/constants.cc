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



#include "parsed_tools/constants.h"

using namespace dealii;

namespace ParsedTools
{
  Constants::Constants(
    const std::string              &section_name,
    const std::vector<std::string> &names,
    const std::vector<double>      &default_values,
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



  Constants::operator const std::map<std::string, double> &() const
  {
    return constants;
  }



  const double &
  Constants::operator[](const std::string &key) const
  {
    return constants.at(key);
  }



#ifdef DEAL_II_WITH_SYMENGINE
  /**
   * Return the constants defined in this class as a
   * Differentiation::SD::types::substitution_map.
   */
  Constants::operator const Differentiation::SD::types::substitution_map &()
    const
  {
    constant_substitution_map.clear();
    for (const auto &p : constants)
      constant_substitution_map[Differentiation::SD::Expression(p.first,
                                                                true)] =
        Differentiation::SD::Expression(p.second);
    return constant_substitution_map;
  }
#endif



} // namespace ParsedTools
