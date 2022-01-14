
#include "parsed_tools/function.h"

using namespace dealii;

namespace ParsedTools
{
  template <int dim>
  Function<dim>::Function(const std::string &section_name,
                          const std::string &expression,
                          const std::string &function_description,
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
    , constants(constants)
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
    ParameterAcceptor::prm.add_action(function_description,
                                      [&](const std::string &) { reinit(); });
    leave_my_subsection(ParameterAcceptor::prm);
  }



  template <int dim>
  void
  Function<dim>::reinit()
  {
    this->FunctionParser<dim>::initialize(
      variable_names,
      expression,
      constants,
      Utilities::split_string_list(variable_names, ",").size() > dim);
  }



  template <int dim>
  void
  Function<dim>::update_constants(
    const std::map<std::string, double> &constants)
  {
    this->constants = constants;
    reinit();
  }



  template <int dim>
  void
  Function<dim>::update_expression(const std::string &expression)
  {
    this->expression = expression;
    reinit();
  }

  template class Function<1>;
  template class Function<2>;
  template class Function<3>;
} // namespace ParsedTools
