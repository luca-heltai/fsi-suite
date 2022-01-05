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

using DimSpacedimTypesNoOne = ::testing::Types<TwoTwo, TwoThree, ThreeThree>;

using DimTypes = ::testing::Types<OneOne, TwoTwo, ThreeThree>;

using DimTypesNoOne = ::testing::Types<TwoTwo, ThreeThree>;

template <class DimSpacedim>
class DimSpacedimTester : public ::testing::Test
{
public:
  using Tdim      = typename DimSpacedim::first_type;
  using Tspacedim = typename DimSpacedim::second_type;

  static constexpr unsigned int dim      = Tdim::value;
  static constexpr unsigned int spacedim = Tspacedim::value;

  // Helper function to parse a string into the parameter acceptor
  void
  parse(const std::string &prm_string)
  {
    ParameterAcceptor::prm.parse_input_from_string(prm_string);
    ParameterAcceptor::parse_all_parameters();
  }

  template <class T>
  void
  parse(const std::string &prm_string, T &t)
  {
    t.ParameterAcceptor::enter_my_subsection(ParameterAcceptor::prm);
    ParameterAcceptor::prm.parse_input_from_string(prm_string);
    t.ParameterAcceptor::leave_my_subsection(ParameterAcceptor::prm);
    ParameterAcceptor::parse_all_parameters();
  }

  std::string
  id() const
  {
    return std::to_string(dim) + "D-" + std::to_string(spacedim) + "D";
  }
};


template <class DimSpacedim>
using DimSpacedimTesterNoOne = DimSpacedimTester<DimSpacedim>;

template <class DimSpacedim>
using DimTester = DimSpacedimTester<DimSpacedim>;

template <class DimSpacedim>
using DimTesterNoOne = DimSpacedimTester<DimSpacedim>;

TYPED_TEST_CASE(DimSpacedimTester, DimSpacedimTypes);
TYPED_TEST_CASE(DimTester, DimTypes);
TYPED_TEST_CASE(DimSpacedimTesterNoOne, DimSpacedimTypesNoOne);
TYPED_TEST_CASE(DimTesterNoOne, DimTypesNoOne);

#endif // fsi_dim_spacedim_tester_h
