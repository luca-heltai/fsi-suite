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
#include "parsed_tools/non_matching_coupling.h"

#include <deal.II/base/quadrature_selector.h>

#include "lac.h"
#include "parsed_tools/enum.h"

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  NonMatchingCoupling<dim, spacedim>::NonMatchingCoupling(
    const std::string &          section_name,
    const dealii::ComponentMask &embedded_mask,
    const dealii::ComponentMask &space_mask,
    const CouplingType           coupling_type,
    const std::string &          quadrature_type,
    const unsigned int           quadrature_order,
    const unsigned int           quadrature_repetitions)
    : ParameterAcceptor(section_name)
    , embedded_mask(embedded_mask)
    , space_mask(space_mask)
    , coupling_type(coupling_type)
    , embedded_quadrature_type(quadrature_type)
    , quadrature_order(quadrature_order)
    , embedded_quadrature_repetitions(quadrature_repetitions)
  {
    add_parameter("Coupling type", this->coupling_type);

    add_parameter("Embedded quadrature type",
                  this->embedded_quadrature_type,
                  "",
                  this->prm,
                  Patterns::Selection(
                    QuadratureSelector<dim>::get_quadrature_names()));

    add_parameter("Embedded quadrature order", this->quadrature_order);

    add_parameter("Embedded quadrature retpetitions",
                  this->embedded_quadrature_repetitions);
  }



  template <int dim, int spacedim>
  void
  NonMatchingCoupling<dim, spacedim>::initialize(
    const GridTools::Cache<spacedim, spacedim> &space_cache,
    const DoFHandler<spacedim, spacedim> &      space_dh,
    const AffineConstraints<double> &           space_constraints,
    const GridTools::Cache<dim, spacedim> &     embedded_cache,
    const DoFHandler<dim, spacedim> &           embedded_dh,
    const AffineConstraints<double> &           embedded_constraints)
  {
    this->space_cache          = &space_cache;
    this->space_dh             = &space_dh;
    this->space_constraints    = &space_constraints;
    this->embedded_cache       = &embedded_cache;
    this->embedded_dh          = &embedded_dh;
    this->embedded_constraints = &embedded_constraints;

    embedded_quadrature =
      QIterated<dim>(QuadratureSelector<1>(this->embedded_quadrature_type,
                                           this->quadrature_order),
                     this->embedded_quadrature_repetitions);
  }



  template <int dim, int spacedim>
  std::unique_ptr<dealii::DynamicSparsityPattern>
  NonMatchingCoupling<dim, spacedim>::assemble_dynamic_sparsity() const
  {
    Assert(space_dh, ExcNotInitialized());

    if (coupling_type == CouplingType::approximate_L2)
      {
        auto dsp = std::make_unique<dealii::DynamicSparsityPattern>(
          space_dh->n_dofs(), embedded_dh->n_dofs());
        const auto &embedded_mapping = embedded_cache->get_mapping();

        NonMatching::create_coupling_sparsity_pattern(*space_cache,
                                                      *space_dh,
                                                      *embedded_dh,
                                                      embedded_quadrature,
                                                      *dsp,
                                                      *space_constraints,
                                                      space_mask,
                                                      embedded_mask,
                                                      embedded_mapping,
                                                      *embedded_constraints);
        return dsp;
      }
    else
      {
        AssertThrow(
          false, ExcMessage("The requested coupling type is not implemented."));
      }
  }



  template class NonMatchingCoupling<1, 1>;
  template class NonMatchingCoupling<1, 2>;
  template class NonMatchingCoupling<1, 3>;
  template class NonMatchingCoupling<2, 2>;
  template class NonMatchingCoupling<2, 3>;
  template class NonMatchingCoupling<3, 3>;

} // namespace ParsedTools