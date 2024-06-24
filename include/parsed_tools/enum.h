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

#ifndef parsed_tools_enum_h
#define parsed_tools_enum_h

#include <deal.II/base/patterns.h>

#include <deal.II/differentiation/sd.h>

#include "parsed_tools/magic_enum.hpp"

namespace dealii
{
  namespace Patterns
  {
    namespace Tools
    {
      /**
       * Helper class to convert enum types to strings and viceversa.
       */
      template <class T>
      struct Convert<T, typename std::enable_if<std::is_enum<T>::value>::type>
      {
        static std::unique_ptr<Patterns::PatternBase>
        to_pattern()
        {
          const auto               n     = magic_enum::enum_names<T>();
          std::vector<std::string> names = {n.begin(), n.end()};
          const auto               selection =
            Patterns::Tools::Convert<decltype(names)>::to_string(
              names,
              Patterns::List(
                Patterns::Anything(), names.size(), names.size(), "|"));
          // Allow parsing a list of enums, and make bitwise or between them
          return Patterns::List(Patterns::Selection(selection),
                                0,
                                names.size(),
                                "|")
            .clone();
        }

        static std::string
        to_string(const T                     &value,
                  const Patterns::PatternBase &p = *Convert<T>::to_pattern())
        {
          namespace B                     = magic_enum::bitwise_operators;
          const auto               values = magic_enum::enum_values<T>();
          std::vector<std::string> names;
          for (const auto &v : values)
            if (B::operator&(value, v) == v)
              names.push_back(std::string(magic_enum::enum_name(v)));
          return Patterns::Tools::Convert<decltype(names)>::to_string(names, p);
        }

        static T
        to_value(const std::string                   &s,
                 const dealii::Patterns::PatternBase &p = *to_pattern())
        {
          namespace B = magic_enum::bitwise_operators;
          // Make sure we have a valid enum value, or empty value
          AssertThrow(p.match(s), ExcNoMatch(s, p.description()));
          T                        value = T();
          std::vector<std::string> value_names;
          value_names =
            Patterns::Tools::Convert<decltype(value_names)>::to_value(s, p);
          for (const auto &name : value_names)
            {
              auto v = magic_enum::enum_cast<T>(name);
              if (v.has_value())
                value = B::operator|(value, v.value());
            }
          return value;
        }
      };
    } // namespace Tools
  }   // namespace Patterns
} // namespace dealii
#endif
