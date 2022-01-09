
#include "tools/parsed_boundary_conditions.h"

#include "tools/components.h"
#include "tools/grid_info.h"

#ifdef DEAL_II_WITH_SYMENGINE

using namespace dealii;


namespace dealii
{
  namespace Patterns
  {
    const unsigned int UnsignedInteger::min_int_value =
      std::numeric_limits<unsigned int>::min();
    const unsigned int UnsignedInteger::max_int_value =
      std::numeric_limits<unsigned int>::max();

    const char *UnsignedInteger::description_init = "[UnsignedInteger";


    UnsignedInteger::UnsignedInteger(const unsigned int lower_bound,
                                     const unsigned int upper_bound)
      : lower_bound(lower_bound)
      , upper_bound(upper_bound)
    {}


    bool
    UnsignedInteger::match(const std::string &test_string) const
    {
      std::istringstream str(test_string);

      unsigned int i;
      if (!(str >> i))
        return false;

      // if (!has_only_whitespace(str))
      //   return false;
      // check whether valid bounds
      // were specified, and if so
      // enforce their values
      if (lower_bound <= upper_bound)
        return ((lower_bound <= i) && (upper_bound >= i));
      else
        return true;
    }



    std::string
    UnsignedInteger::description(const OutputStyle style) const
    {
      switch (style)
        {
          case Machine:
            {
              // check whether valid bounds
              // were specified, and if so
              // output their values
              if (lower_bound <= upper_bound)
                {
                  std::ostringstream description;

                  description << description_init << " range " << lower_bound
                              << "..." << upper_bound << " (inclusive)]";
                  return description.str();
                }
              else
                // if no bounds were given, then
                // return generic string
                return "[ UnsignedInteger]";
            }
          case Text:
            {
              if (lower_bound <= upper_bound)
                {
                  std::ostringstream description;

                  description << "An unsigned integer n such that "
                              << lower_bound << " <= n <= " << upper_bound;

                  return description.str();
                }
              else
                return "An unsigned integer";
            }
          case LaTeX:
            {
              if (lower_bound <= upper_bound)
                {
                  std::ostringstream description;

                  description << "An unsigned integer @f$n@f$ such that @f$"
                              << lower_bound << "\\leq n \\leq " << upper_bound
                              << "@f$";

                  return description.str();
                }
              else
                return "An unsigned integer";
            }
          default:
            AssertThrow(false, ExcNotImplemented());
        }
      // Should never occur without an exception, but prevent compiler from
      // complaining
      return "";
    }



    std::unique_ptr<PatternBase>
    UnsignedInteger::clone() const
    {
      return std::unique_ptr<PatternBase>(
        new UnsignedInteger(lower_bound, upper_bound));
    }



    std::unique_ptr<UnsignedInteger>
    UnsignedInteger::create(const std::string &description)
    {
      if (description.compare(0,
                              std::strlen(description_init),
                              description_init) == 0)
        {
          std::istringstream is(description);

          if (is.str().size() > strlen(description_init) + 1)
            {
              // TODO: verify that description matches the pattern "^\[
              // UnsignedInteger range \d+\.\.\.\d+\]@f$"
              int lower_bound, upper_bound;

              is.ignore(strlen(description_init) + strlen(" range "));

              if (!(is >> lower_bound))
                return std::make_unique<UnsignedInteger>();

              is.ignore(strlen("..."));

              if (!(is >> upper_bound))
                return std::make_unique<UnsignedInteger>();

              return std::make_unique<UnsignedInteger>(lower_bound,
                                                       upper_bound);
            }
          else
            return std::make_unique<UnsignedInteger>();
        }
      else
        return std::unique_ptr<UnsignedInteger>();
    }
  } // namespace Patterns
} // namespace dealii



namespace Tools
{
  template <int spacedim>
  ParsedBoundaryConditions<spacedim>::ParsedBoundaryConditions(
    const std::string &                                      section_name,
    const std::string &                                      component_names,
    const std::vector<std::set<dealii::types::boundary_id>> &ids,
    const std::vector<std::string> &          selected_components,
    const std::vector<BoundaryConditionType> &bc_type,
    const std::vector<std::string> &          expressions)
    : ParameterAcceptor(section_name)
    , component_names(component_names)
    , n_components(Components::n_components(component_names))
    , ids(ids)
    , selected_components(selected_components)
    , bc_type(bc_type)
    , expressions(expressions)
    , grid_info(2)
  {
    Patterns::List expr_pattern(
      Patterns::List(Patterns::Anything(), 0, n_components, ";"),
      0,
      Patterns::List::max_int_value,
      "%");

    Patterns::List comp_pattern(
      Patterns::List(Patterns::Anything(), 0, n_components, ","),
      0,
      Patterns::List::max_int_value,
      ";");

    Patterns::List id_pattern(Patterns::List(Patterns::UnsignedInteger(),
                                             0,
                                             Patterns::List::max_int_value,
                                             ","),
                              0,
                              Patterns::List::max_int_value,
                              ";");

    add_parameter("Boundary id sets (" + component_names + ")",
                  this->ids,
                  "",
                  this->prm,
                  id_pattern);

    add_parameter("Selected components (" + component_names + ")",
                  this->selected_components,
                  "",
                  this->prm,
                  comp_pattern);

    add_parameter("Boundary condition types (" + component_names + ")",
                  this->bc_type);

    add_parameter("Expressions (" + component_names + ")",
                  this->expressions,
                  "",
                  this->prm,
                  expr_pattern);

    this->parse_parameters_call_back.connect([&]() {
      // Parse expressions into functions.
      functions.clear();
      for (const auto &exp : this->expressions)
        functions.emplace_back(
          std::make_unique<dealii::Functions::SymbolicFunction<spacedim>>(exp));

      // Parse components into masks and types
      masks.clear();
      types.clear();
      for (const auto &comp : this->selected_components)
        {
          masks.push_back(Components::mask(this->component_names, comp));
          types.push_back(Components::type(this->component_names, comp));
        }

      // Check that everything is consistent
      check_consistency();
    });
  }



  template <int spacedim>
  void
  ParsedBoundaryConditions<spacedim>::check_consistency() const
  {
    n_boundary_conditions = ids.size();
    AssertThrow(n_boundary_conditions == bc_type.size(),
                ExcMessage("The number of boundary ids must be equal to the "
                           "number of boundary condition types."));
    AssertThrow(n_boundary_conditions == expressions.size(),
                ExcMessage("The number of boundary ids must be equal to the "
                           "number of expressions."));
    AssertThrow(n_boundary_conditions == selected_components.size(),
                ExcMessage("The number of boundary ids "
                           "must be equal to the number "
                           "of selected components."));

    std::set<types::boundary_id> all_ids;
    unsigned int                 n_ids = 0;
    for (const auto &id : ids)
      {
        all_ids.insert(id.begin(), id.end());
        n_ids += id.size();
      }

    // Special meaning of boundary id numbers::invalid_boundary_id:
    if (all_ids.find(numbers::invalid_boundary_id) != all_ids.end())
      {
        AssertThrow(all_ids.size() == 1,
                    ExcMessage("If you use the boundary id -1, then no other "
                               "boundary ids can be specified"));
        AssertThrow(n_boundary_conditions == 1,
                    ExcMessage("Only one BoundaryCondition can be specified "
                               "with the special boundary id -1"));
      }
    else
      {
        AssertThrow(all_ids.size() == n_ids,
                    ExcMessage("You specified the same "
                               " boundary id more than once "
                               "in two different boundary conditions"));
        // We check consistency with the triangulation
        if (grid_info.n_active_cells > 0)
          {
            AssertThrow(all_ids.size() == grid_info.boundary_ids.size(),
                        ExcMessage("The number of boundary ids specified in "
                                   "the input file does not match the number "
                                   "of boundary ids in the triangulation"));
          }
      }

    // Now check that the types are valid in this dimension
    AssertDimension(n_boundary_conditions, types.size());
    AssertDimension(n_boundary_conditions, masks.size());
    for (unsigned int i = 0; i < n_boundary_conditions; ++i)
      {
        if (types[i] == Components::Type::vector ||
            types[i] == Components::Type::normal ||
            types[i] == Components::Type::tangential)
          {
            AssertThrow(masks[i].n_selected_components() == spacedim,
                        ExcDimensionMismatch(masks[i].n_selected_components(),
                                             spacedim));
          }
        if (types[i] == Components::Type::normal ||
            types[i] == Components::Type::tangential)
          {
            AssertThrow(functions[i]->n_components == spacedim,
                        ExcDimensionMismatch(functions[i]->n_components,
                                             spacedim));
          }
        else
          {
            // In all other cases, it is the mask that determines the number
            // of of components, but the function needs to be of the correct
            // dimension
            AssertThrow(functions[i]->n_components == n_components,
                        ExcDimensionMismatch(functions[i]->n_components,
                                             n_components));
          }
      }
  }



  template <int spacedim>
  void
  ParsedBoundaryConditions<spacedim>::update_user_substitution_map(
    const dealii::Differentiation::SD::types::substitution_map
      &substitution_map)
  {
    for (auto &f : functions)
      f->update_user_substitution_map(substitution_map);
  }



  template <int spacedim>
  void
  ParsedBoundaryConditions<spacedim>::set_additional_function_arguments(
    const Differentiation::SD::types::substitution_map &arguments)
  {
    for (auto &f : functions)
      f->set_additional_function_arguments(arguments);
  }



  template <int spacedim>
  void
  ParsedBoundaryConditions<spacedim>::set_time(const double &time)
  {
    for (auto &f : functions)
      f->set_time(time);
  }



  template class ParsedBoundaryConditions<1>;
  template class ParsedBoundaryConditions<2>;
  template class ParsedBoundaryConditions<3>;
} // namespace Tools
#endif
