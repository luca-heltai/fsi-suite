#ifndef patterns_unsigned_int_h
#define patterns_unsigned_int_h

#include <deal.II/base/config.h>

#include <deal.II/base/patterns.h>

namespace dealii
{
  namespace Patterns
  {
    class UnsignedInteger : public PatternBase
    {
    public:
      static const unsigned int min_int_value;

      static const unsigned int max_int_value;

      UnsignedInteger(const unsigned int lower_bound = min_int_value,
                      const unsigned int upper_bound = max_int_value);

      virtual bool
      match(const std::string &test_string) const override;

      virtual std::string
      description(const OutputStyle style = Machine) const override;

      virtual std::unique_ptr<PatternBase>
      clone() const override;

      static std::unique_ptr<UnsignedInteger>
      create(const std::string &description);

    private:
      const unsigned int lower_bound;

      const unsigned int upper_bound;

      static const char *description_init;
    };
  } // namespace Patterns
} // namespace dealii

#endif