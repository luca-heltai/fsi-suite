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

#ifndef projection_operator_h
#define projection_operator_h

#include <deal.II/base/config.h>

#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
namespace dealii
{
  /**
   * Construct a LinearOperator object that projects a vector onto a basis.
   *
   * This function can be used to perform model order reduction, i.e. to project
   * a large dimensional vector onto a smaller dimensional subspace. A typical
   * usage of this class is the following:
   *
   * @code{.cpp}
   * // Build a reduced vector space, i.e., a VectorType (either serial or
   * // parallel) that is an element of the Range space of the LinearOperator
   * VectorType exemplar_range(...);
   *
   * // Initialize a vector of references to the locally owned basis vectors
   * // onto which we want to project.
   * std::vector<std::reference_wrapper<VectorType>> basis(...);
   *
   * P = projection_operator(exemplar_range, basis);
   * Pt = transpose_operator(P);
   *
   * // Project a vector onto the reduced space
   * VectorType elements = P * v;
   *
   * // If the basis was orthogonal, then we could compute an orthogonal
   * // projection of v onto the subspace identified by the basis as
   * VectorType  v_reduced = Pt * v;
   *
   * // The above could also be achieved with the following:
   * VectorType  v_reduced = (Pt*P) * v;
   *
   * // If we want to project an operator A onto the reduced space, we could do
   * // the following:
   * auto A_reduced = Pt*A*P;
   * @endcode
   *
   * @param range_exemplar An exmeplar vector of the range space. We'll make a
   * few copies of this vector to allow the operator to be used with
   * intermediate storage.
   * @param local_basis A vector of references to the locally owned basis
   * vectors. These vectors must be ordered according to the IndexSet
   * Range::locally_owned_elements() of the range space.
   * @param domain_exemplar If used in parallel, it may happen that some
   * processes do not own any index of the Range space. In this case, you need
   * to provide a pointer to a Domain vector to be used as an exemplar vector.
   *
   * @return LinearOperator<Range, Domain, Payload>
   */
  template <
    typename Range,
    typename Domain,
    typename Payload = internal::LinearOperatorImplementation::EmptyPayload>
  LinearOperator<Range, Domain, Payload>
  projection_operator(
    const Range                                             &range_exemplar,
    const std::vector<std::reference_wrapper<const Domain>> &local_basis,
    const Domain  *domain_exemplar = nullptr,
    const Payload &payload         = Payload())
  {
    LinearOperator<Range, Domain, Payload> linear_operator(payload);
    linear_operator.vmult = [range_exemplar, local_basis](Range        &dst,
                                                          const Domain &src) {
      static const auto id = range_exemplar.locally_owned_elements();
      AssertDimension(local_basis.size(), id.n_elements());
      unsigned int i = 0;
      for (const auto j : id)
        dst[j] = local_basis[i++].get() * src;
    };

    linear_operator.vmult_add = [range_exemplar,
                                 local_basis](Range &dst, const Domain &src) {
      static const auto id = range_exemplar.locally_owned_elements();
      AssertDimension(local_basis.size(), id.n_elements());
      unsigned int i = 0;
      for (const auto j : id)
        dst[j] += local_basis[i++].get() * src;
    };

    linear_operator.Tvmult = [range_exemplar, local_basis](Domain      &dst,
                                                           const Range &src) {
      static const auto id = range_exemplar.locally_owned_elements();
      AssertDimension(local_basis.size(), id.n_elements());
      dst            = 0;
      unsigned int i = 0;
      for (const auto j : id)
        dst.sadd(1.0, src[j], local_basis[i++]);
      dst.compress(VectorOperation::add);
    };

    linear_operator.Tvmult_add = [range_exemplar,
                                  local_basis](Domain &dst, const Range &src) {
      static const auto id = range_exemplar.locally_owned_elements();
      AssertDimension(local_basis.size(), id.n_elements());
      unsigned int i = 0;
      for (const auto j : id)
        dst.sadd(1.0, src[j], local_basis[i++]);
      dst.compress(VectorOperation::add);
    };

    linear_operator.reinit_range_vector = [range_exemplar](Range &dst,
                                                           bool   fast) {
      dst.reinit(range_exemplar, fast);
    };

    if (domain_exemplar != nullptr)
      linear_operator.reinit_domain_vector = [domain_exemplar](Domain &dst,
                                                               bool    fast) {
        dst.reinit(*domain_exemplar, fast);
      };
    else
      {
        Assert(local_basis.size() > 0,
               ExcMessage("If domain_exemplar is not provided, "
                          "local_basis must contain at least one element."));
        linear_operator.reinit_domain_vector = [local_basis](Domain &dst,
                                                             bool    fast) {
          dst.reinit(local_basis[0].get(), fast);
        };
      }
    return linear_operator;
  }
} // namespace dealii
#endif