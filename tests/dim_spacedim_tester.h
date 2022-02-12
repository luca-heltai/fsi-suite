// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Luca Heltai
//
// This file is part of the FSI-suite platform, based on the deal.II library.
//
// The FSI-suite platform is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3.0 of the License,
// or (at your option) any later version. The full text of the license can be
// found in the file LICENSE at the top level of the FSI-suite platform
// distribution.
//
// ---------------------------------------------------------------------

#ifndef fsi_dim_spacedim_tester_h
#define fsi_dim_spacedim_tester_h

// common definitions used in all the tests

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

/**
 * @brief Wrap a code block with try-catch, handle exceptions thrown, print them
 * into EXCEPT_STREAM and rethrow.
 */
#define PRINT_AND_RETHROW(CODE_BLOCK, EXCEPT_STREAM)                        \
  try                                                                       \
    {                                                                       \
      do                                                                    \
        {                                                                   \
          CODE_BLOCK                                                        \
        }                                                                   \
      while (0);                                                            \
    }                                                                       \
  catch (const std::exception &ex)                                          \
    {                                                                       \
      EXCEPT_STREAM << "std::exception thrown: " << ex.what() << std::endl; \
      throw;                                                                \
    }                                                                       \
  catch (...)                                                               \
    {                                                                       \
      EXCEPT_STREAM << "unknown structure thrown" << std::endl;             \
      throw;                                                                \
    }


/**
 * @brief Wrap a code block with try-catch, handle exceptions thrown, print them
 * into std::cerr and rethrow.
 */
#define PRINT_STDERR_AND_RETHROW(CODE_BLOCK) \
  PRINT_AND_RETHROW(CODE_BLOCK, std::cerr)

#define EXPECT_NO_THROW_PRINT(CODE_BLOCK) \
  EXPECT_NO_THROW(PRINT_STDERR_AND_RETHROW(CODE_BLOCK))

#define ASSERT_NO_THROW_PRINT(CODE_BLOCK) \
  ASSERT_NO_THROW(PRINT_STDERR_AND_RETHROW(CODE_BLOCK))


using namespace dealii;


template <class T>
void
parse(const std::string &prm_string, T &t)
{
  t.ParameterAcceptor::enter_my_subsection(ParameterAcceptor::prm);
  ParameterAcceptor::prm.parse_input_from_string(prm_string);
  t.ParameterAcceptor::leave_my_subsection(ParameterAcceptor::prm);
  ParameterAcceptor::parse_all_parameters();
}

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
  id(const std::string &suffix = "") const
  {
    const testing::TestInfo *const test_info =
      testing::UnitTest::GetInstance()->current_test_info();

    return std::string("/") + test_info->test_suite_name() + "/" +
           std::to_string(dim) + "D-" + std::to_string(spacedim) + "D/" +
           suffix;
  }
};


template <class DimSpacedim>
using DimSpacedimTesterNoOne = DimSpacedimTester<DimSpacedim>;

template <class DimSpacedim>
using DimTester = DimSpacedimTester<DimSpacedim>;

template <class DimSpacedim>
using DimTesterNoOne = DimSpacedimTester<DimSpacedim>;

using OneTester      = DimSpacedimTester<OneOne>;
using OneTwoTester   = DimSpacedimTester<OneTwo>;
using OneThreeTester = DimSpacedimTester<OneThree>;
using TwoTester      = DimSpacedimTester<TwoTwo>;
using ThreeTester    = DimSpacedimTester<ThreeThree>;
using TwoThreeTester = DimSpacedimTester<TwoThree>;
using ThreeTester    = DimSpacedimTester<ThreeThree>;


TYPED_TEST_CASE(DimSpacedimTester, DimSpacedimTypes);
TYPED_TEST_CASE(DimTester, DimTypes);
TYPED_TEST_CASE(DimSpacedimTesterNoOne, DimSpacedimTypesNoOne);
TYPED_TEST_CASE(DimTesterNoOne, DimTypesNoOne);

#endif // fsi_dim_spacedim_tester_h
