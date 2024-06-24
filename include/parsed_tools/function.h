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

#ifndef parsed_tools_function_h
#define parsed_tools_function_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

namespace ParsedTools
{
  /**
   * A wrapper for the dealii::FunctionParser class.
   */
  template <int dim>
  class Function : public dealii::ParameterAcceptor,
                   public dealii::FunctionParser<dim>
  {
  public:
    /**
     * Build a Function based on ParameterAcceptor.
     *
     * An example usage of this class is the following:
     *
     * @code
     * ParsedTools::Function<dim> my_function("Rhs function","2*x+y");
     * auto value = my_function.value(some_point);
     * @endcode
     *
     * The above snippet of code will delcare the ParameterAcceptor::prm
     * ParameterHandler with the following entries:
     *
     * @code{.sh}
     * subsection RHS function
     *   set Function expression = "2*x+y"
     * end
     * @endcode
     *
     * @param section_name The name of ParameterAcceptor section
     * @param expression The expression of the function
     * @param function_description How the expression is declared in the
     *        parameter file
     * @param constants A map of constants we can use in the function
     *        expression
     * @param variable_names Comma separated names of the independent dvariables
     *        in the expression. Either dim or dim+1 (for time dependent
     *        problems) names must be given.
     * @param h The step size for finite difference approximation of derivatives
     */
    Function(const std::string &section_name         = "",
             const std::string &expression           = "0",
             const std::string &function_description = "Function expression",
             const std::map<std::string, double> &constants = {},
             const std::string                   &variable_names =
               dealii::FunctionParser<dim>::default_variable_names() + ",t",
             const double h = 1e-8);

    /**
     * Reinitialize the function with the new constants.
     */
    void
    update_constants(const std::map<std::string, double> &constants);

    /**
     * Reinitialize the function with the new expression.
     */
    void
    update_expression(const std::string &expr);

  private:
    /**
     * Reset the Function object using the expression, the constants, and
     * the variables stored in this class.
     */
    void
    reinit();

    /**
     * The actual FunctionParser expression.
     */
    std::string expression;

    /**
     * Constants that can be used in the expression.
     */
    std::map<std::string, double> constants;

    /**
     * Keep variable names around to re-initialize the FunctionParser class.
     * */
    const std::string variable_names;
  };
} // namespace ParsedTools
#endif