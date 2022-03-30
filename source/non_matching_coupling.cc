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

#include <boost/geometry.hpp>

#include "lac.h"
#include "parsed_tools/enum.h"

using namespace magic_enum::bitwise_operators;

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  NonMatchingCoupling<dim, spacedim>::NonMatchingCoupling(
    const std::string &                                    section_name,
    const dealii::ComponentMask &                          embedded_mask,
    const dealii::ComponentMask &                          space_mask,
    const NonMatchingCoupling<dim, spacedim>::CouplingType coupling_type,
    const NonMatchingCoupling<dim, spacedim>::RefinementStrategy
                       refinement_strategy,
    const unsigned int space_pre_refinement,
    const unsigned int embedded_post_refinement,
    const std::string &quadrature_type,
    const unsigned int quadrature_order,
    const unsigned int quadrature_repetitions)
    : ParameterAcceptor(section_name)
    , embedded_mask(embedded_mask)
    , space_mask(space_mask)
    , coupling_type(coupling_type)
    , refinement_strategy(refinement_strategy)
    , space_pre_refinement(space_pre_refinement)
    , embedded_post_refinement(embedded_post_refinement)
    , embedded_quadrature_type(quadrature_type)
    , quadrature_order(quadrature_order)
    , embedded_quadrature_repetitions(quadrature_repetitions)
  {
    add_parameter("Coupling type", this->coupling_type);

    add_parameter("Refinement strategy", this->refinement_strategy);

    add_parameter("Space pre-refinement", this->space_pre_refinement);

    add_parameter("Embedded post-refinement", this->embedded_post_refinement);

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
  typename NonMatchingCoupling<dim, spacedim>::CouplingType
  NonMatchingCoupling<dim, spacedim>::get_coupling_type() const
  {
    return this->coupling_type;
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


  template <int dim, int spacedim>
  void
  NonMatchingCoupling<dim, spacedim>::adjust_grid_refinements(
    Triangulation<spacedim, spacedim> &space_tria,
    Triangulation<dim, spacedim> &     embedded_tria,
    const bool                         apply_delta_refinements) const
  {
    Assert(space_dh, ExcNotInitialized());
    Assert(&embedded_tria == &embedded_dh->get_triangulation(),
           ExcMessage(
             "The passed embedded triangulation must be the same as the "
             "one used by the embedded DoFHandler with which you "
             "initialized this class."));
    Assert(&space_tria == &space_dh->get_triangulation(),
           ExcMessage("The passed space triangulation must be the same as the "
                      "one used by the space DoFHandler with which you "
                      "initialized this class."));

    namespace bgi = boost::geometry::index;

    auto refine = [&]() {
      bool done = false;

      double min_embedded = 1e10;
      double max_embedded = 0;
      double min_space    = 1e10;
      double max_space    = 0;

      while (done == false)
        {
          // Bounding boxes of the space grid
          const auto &tree =
            space_cache->get_locally_owned_cell_bounding_boxes_rtree();

          // Bounding boxes of the embedded grid
          const auto &embedded_tree =
            embedded_cache->get_locally_owned_cell_bounding_boxes_rtree();

          // Let's check all cells whose bounding box contains an embedded
          // bounding box
          done = true;

          const bool use_space =
            ((this->refinement_strategy & RefinementStrategy::refine_space) ==
             RefinementStrategy::refine_space);

          const bool use_embedded = ((this->refinement_strategy &
                                      RefinementStrategy::refine_embedded) ==
                                     RefinementStrategy::refine_embedded);
          AssertThrow(!(use_embedded && use_space),
                      ExcMessage("You can't refine both the embedded and "
                                 "the space grid at the same time."));

          for (const auto &[embedded_box, embedded_cell] : embedded_tree)
            {
              const auto &[p1, p2] = embedded_box.get_boundary_points();
              const auto diameter  = p1.distance(p2);
              min_embedded         = std::min(min_embedded, diameter);
              max_embedded         = std::max(max_embedded, diameter);

              for (const auto &[space_box, space_cell] :
                   tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
                {
                  const auto &[sp1, sp2]    = space_box.get_boundary_points();
                  const auto space_diameter = sp1.distance(sp2);
                  min_space = std::min(min_space, space_diameter);
                  max_space = std::max(max_space, space_diameter);

                  if (use_embedded && space_diameter < diameter)
                    {
                      embedded_cell->set_refine_flag();
                      done = false;
                    }
                  if (use_space && diameter < space_diameter)
                    {
                      space_cell->set_refine_flag();
                      done = false;
                    }
                }
            }
          if (done == false)
            {
              if (use_embedded)
                {
                  // Compute again the embedded displacement grid
                  embedded_tria.execute_coarsening_and_refinement();
                  embedded_post_refinemnt_signal();
                }
              if (use_space)
                {
                  // Compute again the embedded displacement grid
                  space_tria.execute_coarsening_and_refinement();
                  space_post_refinemnt_signal();
                }
            }
        }
      return std::make_tuple(min_space, max_space, min_embedded, max_embedded);
    };

    // Do the refinement loop once, to make sure we satisfy our criterions
    refine();

    // Pre refine the space grid according to the delta refinement
    if (apply_delta_refinements && space_pre_refinement != 0)
      for (unsigned int i = 0; i < space_pre_refinement; ++i)
        {
          const auto &tree =
            space_cache->get_locally_owned_cell_bounding_boxes_rtree();

          const auto &embedded_tree =
            embedded_cache->get_locally_owned_cell_bounding_boxes_rtree();

          for (const auto &[embedded_box, embedded_cell] : embedded_tree)
            for (const auto &[space_box, space_cell] :
                 tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
              space_cell->set_refine_flag();
          space_tria.execute_coarsening_and_refinement();

          // Make sure we signal post refinement on the space grid
          space_post_refinemnt_signal();

          // Make sure again we satisfy our criterion after the space refinement
          refine();
        }

    // Post refinement on embedded grid is easy
    if (apply_delta_refinements && embedded_post_refinement != 0)
      {
        embedded_tria.refine_global(embedded_post_refinement);
        embedded_post_refinemnt_signal();
      }

    // Check once again we satisfy our criterion, and record min/max
    const auto [sm, sM, em, eM] = refine();

    deallog << "Space local min/max diameters   : " << sm << "/" << sM
            << std::endl
            << "Embedded space min/max diameters: " << em << "/" << eM
            << std::endl;
  }



  template class NonMatchingCoupling<1, 1>;
  template class NonMatchingCoupling<1, 2>;
  template class NonMatchingCoupling<1, 3>;
  template class NonMatchingCoupling<2, 2>;
  template class NonMatchingCoupling<2, 3>;
  template class NonMatchingCoupling<3, 3>;

} // namespace ParsedTools