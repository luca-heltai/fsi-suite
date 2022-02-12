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

#ifndef parsed_tools_patterns_h
#define parsed_tools_patterns_h

#include <deal.II/base/config.h>

#include <deal.II/base/patterns.h>

namespace dealii
{
  namespace Patterns
  {
    class UnsignedInteger : public PatternBase
    {
    public:
      static const unsigned int min_int_value;

      static const unsigned int max_int_value;

      UnsignedInteger(const unsigned int lower_bound = min_int_value,
                      const unsigned int upper_bound = max_int_value);

      virtual bool
      match(const std::string &test_string) const override;

      virtual std::string
      description(const OutputStyle style = Machine) const override;

      virtual std::unique_ptr<PatternBase>
      clone() const override;

      static std::unique_ptr<UnsignedInteger>
      create(const std::string &description);

    private:
      const unsigned int lower_bound;

      const unsigned int upper_bound;

      static const char *description_init;
    };
  } // namespace Patterns
} // namespace dealii

#endif