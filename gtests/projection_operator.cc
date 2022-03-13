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

#include <deal.II/base/config.h>

#include "projection_operator.h"

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "tests.h"

using namespace dealii;

TEST(ProjectionOperator, ProjectionOnSubspace)
{
  initlog();
  // Orthogonal projection on the first M basis
  const unsigned int          N = 5;
  const unsigned int          M = 3;
  std::vector<Vector<double>> v(M, Vector<double>(N));
  std::vector<std::reference_wrapper<const Vector<double>>> basis(v.begin(),
                                                                  v.end());
  for (unsigned int i = 0; i < M; ++i)
    {
      v[i][i]  = 1;
      basis[i] = std::cref(v[i]);
      deallog << "Basis " << i << ": " << v[i] << std::endl;
    }

  Vector<double> w(M);

  auto C  = projection_operator(w, basis);
  auto Ct = transpose_operator(C);

  auto Iw = C * Ct;
  auto Pw = Ct * C;

  // Try it out: w in R^M
  for (unsigned int i = 0; i < M; ++i)
    w[i] = 2 * i + 1;
  Vector<double> x(N);

  // x in R^N
  for (unsigned int i = 0; i < N; ++i)
    x[i] = 2 * i + 1;

  deallog << "w: " << w << std::endl;
  deallog << "x: " << x << std::endl;
  double w_norm = w.l2_norm();
  double x_norm = x.l2_norm();

  Vector<double> y = Ct * w; // in R^N
  deallog << "Ct*w =  " << y << std::endl;

  Vector<double> p = C * x; // in R^M
  deallog << "C*x =   " << p << std::endl;
  ASSERT_DOUBLE_EQ(p.l2_norm(), w_norm); // Same as w

  Vector<double> z = Iw * w; // in R^M
  deallog << "(C*Ct)*w =  " << z << std::endl;
  ASSERT_DOUBLE_EQ(z.l2_norm(), w_norm); // Same norm as w
  z -= w;
  // They should be actually equal
  ASSERT_DOUBLE_EQ(z.l2_norm(), 0.0); // Same as w

  Vector<double> q = Pw * x; // in R^N
  deallog << "(Ct*C)*x = " << q << std::endl;
  // again, q should have the same norm as w
  ASSERT_DOUBLE_EQ(p.l2_norm(), w_norm);
}
