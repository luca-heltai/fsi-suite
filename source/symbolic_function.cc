
#include "parsed_tools/symbolic_function.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

namespace ParsedTools
{
  template <int dim>
  SymbolicFunction<dim>::SymbolicFunction(
    const std::string &section_name,
    const std::string &expression,
    const std::string &function_description)
    : ParameterAcceptor(section_name)
    , symbolic_function(
        std::make_unique<dealii::Functions::SymbolicFunction<dim>>(expression))
    , n_components(symbolic_function->n_components)
  {
    add_parameter(function_description, this->symbolic_function);
    enter_my_subsection(ParameterAcceptor::prm);
    // Make sure we have the correct number of components when parsing a new
    // expression
    ParameterAcceptor::prm.add_action(
      function_description, [&](const std::string &) {
        AssertThrow(n_components == symbolic_function->n_components,
                    ExcMessage("Invalid number of components."));
      });
    leave_my_subsection(ParameterAcceptor::prm);
  }

  template class SymbolicFunction<1>;
  template class SymbolicFunction<2>;
  template class SymbolicFunction<3>;
} // namespace ParsedTools

#endif