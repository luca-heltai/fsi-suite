#ifndef tools_parsed_function_h
#define tools_parsed_function_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

namespace Tools
{
  /**
   * A wrapper for the dealii::FunctionParser class.
   */
  template <int dim>
  class ParsedFunction : public dealii::ParameterAcceptor,
                         public dealii::FunctionParser<dim>
  {
  public:
    /**
     * Build a ParameterAcceptor based ParsedFunction.
     *
     * Notice that the @p constants parameter must persist as long as this class
     * is alive, since the we store a reference to it.
     *
     * An example usage of this class is the following:
     *
     * @code
     * Tools::ParsedFunction my_fun
     * @endcode
     *
     * The above snippet of code will delcare the ParameterAcceptor::prm
     * ParameterHandler with the following entries:
     *
     * @code{.sh}
     *
     * @endcode
     */
    ParsedFunction(
      const std::string &section_name                = "",
      const std::string &expression                  = "",
      const std::string &function_description        = "Function expression",
      const std::map<std::string, double> &constants = {},
      const std::string &                  variable_names =
        dealii::FunctionParser<dim>::default_variable_names() + ",t",
      const double h = 1e-8);

  private:
    /**
     * The actual FunctionParser expression.
     */
    std::string expression;

    /**
     * A local copy of the object passed at constructor, used as a reference for
     * the actual constants when the constructor is called with a
     * rvalue_reference.
     */
    const std::map<std::string, double> local_constants;

    /**
     * Constants that can be used in the expression.
     */
    const std::map<std::string, double> &constants;

    /**
     * Keep variable names around to re-initialize the FunctionParser class.
     * */
    const std::string variable_names;
  };
} // namespace Tools
#endif