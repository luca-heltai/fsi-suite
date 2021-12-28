#ifndef fsi_dim_spacedim_tester_h
#define fsi_dim_spacedim_tester_h

// common definitions used in all the tests

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

using namespace dealii;

using One   = std::integral_constant<int, 1>;
using Two   = std::integral_constant<int, 2>;
using Three = std::integral_constant<int, 3>;

struct OneOne : public std::pair<One, One>
{};
struct OneTwo : public std::pair<One, Two>
{};
struct OneThree : public std::pair<One, Three>
{};
struct TwoTwo : public std::pair<Two, Two>
{};
struct TwoThree : public std::pair<Two, Three>
{};
struct ThreeThree : public std::pair<Three, Three>
{};

using DimSpacedimTypes =
  ::testing::Types<OneOne, OneTwo, OneThree, TwoTwo, TwoThree, ThreeThree>;

template <class DimSpacedim>
class DST : public ::testing::Test
{
public:
  using Tdim      = typename DimSpacedim::first_type;
  using Tspacedim = typename DimSpacedim::second_type;

  static constexpr unsigned int dim      = Tdim::value;
  static constexpr unsigned int spacedim = Tspacedim::value;

  void
  parse(const std::string &prm_string) const
  {
    ParameterAcceptor::prm.parse_input_from_string(prm_string);
    ParameterAcceptor::parse_all_parameters();
  }
};

TYPED_TEST_CASE(DST, DimSpacedimTypes);

#endif // dealii_tests_h
