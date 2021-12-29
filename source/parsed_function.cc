
#include "tools/parsed_function.h"

using namespace dealii;

namespace Tools
{
  template <int dim>
  ParsedFunction<dim>::ParsedFunction(
    const std::string &                  section_name,
    const std::string &                  expression,
    const std::string &                  function_description,
    const std::map<std::string, double> &constants,
    const std::string &                  variable_names,
    const double                         h)
    : ParameterAcceptor(section_name)
    , FunctionParser<dim>(
        expression,
        Patterns::Tools::Convert<std::map<std::string, double>>::to_string(
          constants,
          Patterns::Map(Patterns::Anything(),
                        Patterns::Double(),
                        0,
                        Patterns::Map::max_int_value,
                        ",",
                        "=")),
        variable_names,
        h)
    , expression(expression)
    , local_constants(constants)
    , constants(std::is_lvalue_reference<decltype(constants)>::value ?
                  constants :
                  local_constants)
    , variable_names(variable_names)
  {
    std::string doc =
      function_description + ", with input variables (" + variable_names +
      ")." +
      (constants.empty() ? "" : " You can use the following constants: ");
    std::string sep = "";
    for (const auto &constant : constants)
      {
        doc += sep + constant.first;
        sep = ", ";
      }

    add_parameter(function_description, this->expression, doc);

    enter_my_subsection(ParameterAcceptor::prm);
    ParameterAcceptor::prm.add_action(
      function_description, [&](const std::string &expr) {
        this->FunctionParser<dim>::initialize(
          this->variable_names,
          expr,
          this->constants,
          Utilities::split_string_list(this->variable_names, ",").size() > dim);
      });
    leave_my_subsection(ParameterAcceptor::prm);
  }

  template class ParsedFunction<1>;
  template class ParsedFunction<2>;
  template class ParsedFunction<3>;
} // namespace Tools
