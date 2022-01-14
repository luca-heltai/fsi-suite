#ifndef parsed_tools__symbolic_function_h
#define parsed_tools__symbolic_function_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_SYMENGINE

#  include <deal.II/base/parameter_acceptor.h>
#  include <deal.II/base/symbolic_function.h>

namespace dealii
{
  namespace Patterns
  {
    namespace Tools
    {
      /**
       * Helper class to parse symbolic expressions.
       */
      template <>
      struct Convert<Differentiation::SD::Expression>
      {
        /**
         * Shorthand for the type of the parsed expression.
         */
        using T = Differentiation::SD::Expression;

        /**
         * Default pattern for parsing symbolic expressions.
         *
         * Since we don't really have many restrictions, this pattern is simply
         * the Patterns::Anything() pattern.
         */
        static std::unique_ptr<dealii::Patterns::PatternBase>
        to_pattern()
        {
          return Patterns::Anything().clone();
        }

        /**
         * Convert a Differentiation::SD::Expression expression to a string.
         */
        static std::string
        to_string(const T &                            t,
                  const dealii::Patterns::PatternBase &pattern = *to_pattern())
        {
          std::stringstream ss;
          ss << t;
          AssertThrow(
            pattern.match(ss.str()),
            ExcMessage(
              "The expression does not satisfy the requirements of the "
              "pattern."));
          return ss.str();
        }

        /**
         * Convert a string to a Differentiation::SD::Expression expression.
         */
        static T
        to_value(const std::string &                  s,
                 const dealii::Patterns::PatternBase &pattern = *to_pattern())
        {
          AssertThrow(pattern.match(s), ExcMessage("Invalid string."));
          return T(s, true);
        }
      };

      namespace internal
      {
        // Rank of SymbolicFunction
        template <int dim>
        struct RankInfo<
          std::unique_ptr<dealii::Functions::SymbolicFunction<dim>>>
        {
          static constexpr int list_rank = 1;
          static constexpr int map_rank  = 0;
        };
      } // namespace internal

      /**
       * @brief Instruct deal.II on how to convert a SymbolicFunctions
       * to a string.
       */
      template <int dim>
      struct Convert<std::unique_ptr<dealii::Functions::SymbolicFunction<dim>>>
      {
        /**
         * Shorthand for the type of the parsed expression.
         */
        using T = std::unique_ptr<dealii::Functions::SymbolicFunction<dim>>;

        /**
         * Pattern for a SymbolicFunction.
         */
        static std::unique_ptr<dealii::Patterns::PatternBase>
        to_pattern()
        {
          return Patterns::List(Patterns::Anything(),
                                0,
                                dealii::Patterns::List::max_int_value,
                                ";")
            .clone();
        }

        /**
         * @brief Convert a SymbolicFunction to a string.
         *
         * @param t The pointer to SymbolicFunction to convert.
         * @param pattern Optional pattern to use.
         * @return std::string The string representation of the SymbolicFunction.
         */
        static std::string
        to_string(const T &                            t,
                  const dealii::Patterns::PatternBase &pattern =
                    *Convert<T>::to_pattern())
        {
          auto expr = t->get_symbolic_function_expressions();
          return Convert<decltype(expr)>::to_string(expr, pattern);
        }

        /**
         * @brief Convert a string to a SymbolicFunction.
         *
         * @param s The string to convert.
         * @param pattern Optional pattern to use.
         * @return T The SymbolicFunction.
         */
        static T
        to_value(const std::string &                  s,
                 const dealii::Patterns::PatternBase &pattern =
                   *Convert<T>::to_pattern())
        {
          AssertThrow(pattern.match(s), ExcMessage("Invalid string."));
          return std::make_unique<dealii::Functions::SymbolicFunction<dim>>(s);
        }
      };
    } // namespace Tools
  }   // namespace Patterns
} // namespace dealii



namespace ParsedTools
{
  /**
   * A wrapper for the dealii::Functions::SymbolicFunction class.
   */
  template <int dim>
  class SymbolicFunction : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Build a SymbolicFunction based on the ParameterAcceptor class.
     *
     * An example usage of this class is the following:
     *
     * @code
     * ParsedTools::SymbolicFunction<dim> my_function("Rhs
     * function","2*x+y"); auto value = my_function().value(some_point);
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
     */
    SymbolicFunction(
      const std::string &section_name         = "",
      const std::string &expression           = "",
      const std::string &function_description = "Function expression");

    /**
     * Act as an actual dealii::Functions::SymbolicFunction, so that we can pass
     * this object to any function that expects a dealii::Functions::Function
     * object.
     */
    operator dealii::Functions::SymbolicFunction<dim> &()
    {
      return *symbolic_function;
    }

    /**
     * Act as an actual dealii::Functions::SymbolicFunction.
     */
    dealii::Functions::SymbolicFunction<dim> &
    operator()()
    {
      return *symbolic_function;
    }

  private:
    /**
     * The actual dealii::Functions::SymbolicFunction object.
     */
    std::unique_ptr<dealii::Functions::SymbolicFunction<dim>> symbolic_function;

    /**
     * The number of components of the function. This number is constructed from
     * the expression given at construction time. If you reinitialize the
     * function, make sure you use the same number of components, or an
     * exception will be thrown.
     */
    const unsigned int n_components;
  };
} // namespace ParsedTools
#endif
#endif