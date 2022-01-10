#include "tools/patterns_unsigned_int.h"

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
