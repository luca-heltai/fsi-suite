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

#ifndef parsed_tools_constants_h
#define parsed_tools_constants_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/differentiation/sd.h>

namespace ParsedTools
{
  /**
   * A wrapper for physical constants to be shared among functions and classes.
   *
   * This class can be used to store physical constants that are used in a
   * simulation, e.g., elastic parameters, densities, etc., and that could be
   * reused when defining Function objects.
   *
   * This class behaves like a map, where the keys are strings and the values
   * are the corresponding values.
   *
   * In addition to constants defined at construction time, this class also
   * provides a list of all mathematical constants defined in the
   * dealii::numbers namespace, i.e., PI, E, LOG2_E, etc.
   *
   * Constants given at construction time takes precedence, so if you use "E" as
   * a constant name, than that value will be used instead of the Nepero
   * constant numbers::E.
   */
  class Constants : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Build a ParameterAcceptor based collection of constants.
     *
     * An example usage of this class is the following:
     *
     * @code
     * ParsedTools::Constants constants("/Physical Constants",
     *                                   {"a", "b", "c"},
     *                                   {1.0, 2.0, 3.0},
     *                                   {"The constant a",
     *                                    "The constant b",
     *                                    "The constant c"});
     *
     *  ParameterAcceptor::initialize();
     *  // Now use c as a map whenever one is needed, i.e.,
     *  const std::map<std::string, double> &c = constants;
     *  c.at("a") == 1.0; // returns true
     *  // or directly as
     *  contants["a"] == 1.0; // returns true
     * @endcode
     *
     * The above snippet of code will delcare the ParameterAcceptor::prm
     * ParameterHandler with the following entries:
     *
     * @code{.sh}
     * subsection Physical Constants
     *   set The constant a (a) = 1.0
     *   set The constant b (b) = 2.0
     *   set The constant c (c) = 3.0
     * end
     * @endcode
     *
     * The @p optional_names_for_parameter_file can be left empty, in which case
     * the names of the constants will be used. In the example above, leaving
     * @p optional_names_for_parameter_file empty would result in the following
     * parameter entries:
     * @code{.sh}
     * subsection Physical Constants
     *   set a = 1.0
     *   set b = 2.0
     *   set c = 3.0
     * end
     * @endcode
     *
     * @param section_name The name of the section in the parameter file.
     * @param names The names of the constants.
     * @param default_values  The default values of the constants.
     * @param optional_names_for_parameter_file Optional names to use instead of
     *        the constant names in the parameter file defintion.
     * @param optional_documentation_strings Optional documentation strings to
     *        use in the parameter file defintion.
     */
    Constants(
      const std::string &             section_name                      = "",
      const std::vector<std::string> &names                             = {},
      const std::vector<double> &     default_values                    = {},
      const std::vector<std::string> &optional_names_for_parameter_file = {},
      const std::vector<std::string> &optional_documentation_strings    = {});

    /**
     * Return the constants defined in this class as a map.
     */
    operator const std::map<std::string, double> &() const;

#ifdef DEAL_II_WITH_SYMENGINE
    /**
     * Return the constants defined in this class as a
     * Differentiation::SD::types::substitution_map.
     */
    operator const dealii::Differentiation::SD::types::substitution_map &()
      const;
#endif

    /**
     * Return the constant associated with the given name.
     *
     * @param key The name of the constant.
     * @return const double& The constant associated with the given name.
     */
    const double &
    operator[](const std::string &key) const;

  private:
    /**
     * The actual constants.
     */
    std::map<std::string, double> constants;


#ifdef DEAL_II_WITH_SYMENGINE
    /**
     * Return the constants defined in this class as a
     * Differentiation::SD::types::substitution_map.
     */
    mutable dealii::Differentiation::SD::types::substitution_map
      constant_substitution_map;
#endif


    /**
     * The documentation string used to parse the constants from thee parameter
     * file.
     */
    std::map<std::string, std::string> constants_parameter_entries;

    /**
     * The documentation of each constant.
     */
    std::map<std::string, std::string> constants_documentation;
  };
} // namespace ParsedTools
#endif