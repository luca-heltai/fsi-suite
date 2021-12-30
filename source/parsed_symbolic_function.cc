
#include "tools/parsed_symbolic_function.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;

namespace Tools
{
  template <int dim>
  ParsedSymbolicFunction<dim>::ParsedSymbolicFunction(
    const std::string &section_name,
    const std::string &expression,
    const std::string &function_description)
    : ParameterAcceptor(section_name)
    , symbolic_function(
        std::make_unique<dealii::Functions::SymbolicFunction<dim>>(expression))
  {
    add_parameter(function_description, this->symbolic_function);
  }

  template class ParsedSymbolicFunction<1>;
  template class ParsedSymbolicFunction<2>;
  template class ParsedSymbolicFunction<3>;
} // namespace Tools

#endif