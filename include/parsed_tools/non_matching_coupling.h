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
#ifndef parsed_tools_non_matching_coupling_h
#define parsed_tools_non_matching_coupling_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/non_matching/coupling.h>

#include "assemble_coupling_mass_matrix_with_exact_intersections.h"
#include "compute_intersections.h"
#include "create_coupling_sparsity_pattern_with_exact_intersections.h"

namespace ParsedTools
{
  /**
   * Wrapper around several functions in the dealii::NonMatching namespace.
   *
   * @tparam dim Dimension of the embedded space
   * @tparam spacedim Dimension of the embedding space
   * @tparam LacType Linear algebra types used to assemble the matrices
   */
  template <int dim, int spacedim>
  class NonMatchingCoupling : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Types of coupling that can used in non-matching coupling.
     */
    enum class CouplingType
    {
      // interpolation = 1 << 0, //<! Use interpolation for the coupling
      approximate_L2 =
        1 << 1, //< Approximate L2-projection, using quadrature formulas on the
                // embedded domain to drive the projection.
                //   approximate_H1 = 1 << 2, //< Approximate H1-projection,
                //   using quadrature
                //                            // formulas on the embedded domain
                //                            to drive the
                //                            // projection.
      exact_L2 = 1 << 3, //< Exact L2-projection, using
                         // quadratures on the
                         // intersection to drive the
                         //  projection
                         //   exact_H1 = 1 << 4,       //< Exact H1-projection,
                         //   using quadratures on the
                         //                            // intersection to drive
                         //                            the projection
                         //
    };

    /**
     * Refinement strategy.
     */
    enum class RefinementStrategy
    {
      //! No refinement
      none = 1 << 1,
      //! Force dealii::space grid to be locally smaller than embedded grid
      refine_space = 1 << 2,
      //! Force embedded grid to be locally smaller than embedded grid
      refine_embedded = 1 << 3,
    };

    /**
     * Constructor.
     */
    NonMatchingCoupling(
      const std::string &          section_name  = "",
      const dealii::ComponentMask &embedded_mask = dealii::ComponentMask(),
      const dealii::ComponentMask &space_mask    = dealii::ComponentMask(),
      const CouplingType           coupling_type = CouplingType::approximate_L2,
      const RefinementStrategy     refinement_strategy =
        RefinementStrategy::refine_embedded,
      const unsigned int space_pre_refinement     = 0,
      const unsigned int embedded_post_refinement = 0,
      const std::string &quadrature_type          = "gauss",
      const unsigned int quadrature_order         = 2,
      const unsigned int quadrature_repetitions   = 1,
      double             quadrature_tolerance     = 1e-9);


    /**
     * Initialize the class and return a linear operator representing the
     * coupling matrix.
     */
    void
    initialize(const dealii::GridTools::Cache<spacedim, spacedim> &space_cache,
               const dealii::DoFHandler<spacedim, spacedim> &      space_dh,
               const dealii::AffineConstraints<double> &      space_constraints,
               const dealii::GridTools::Cache<dim, spacedim> &embedded_cache,
               const dealii::DoFHandler<dim, spacedim> &      embedded_dh,
               const dealii::AffineConstraints<double> &embedded_constraints);

    /**
     * Build the coupling sparsity pattern.
     *
     * @tparam SparsityType
     * @param sparsity_pattern
     */
    template <typename SparsityType>
    void
    assemble_sparsity(SparsityType &sparsity_pattern) const;

    /**
     * Assemble the coupling matrix.
     */
    template <typename MatrixType>
    void
    assemble_matrix(MatrixType &matrix) const;

    /**
     * @brief Get the coupling type object
     *
     * @return const CouplingType
     */
    CouplingType
    get_coupling_type() const;

    /**
     * Adjust grid refinements according to the selected strategy.
     */
    void
    adjust_grid_refinements(
      dealii::Triangulation<spacedim, spacedim> &space_tria,
      dealii::Triangulation<dim, spacedim> &     embedded_tria,
      const bool apply_delta_refinements = true) const;

    /**
     * Signals that embedded grid has been just refined.
     */
    boost::signals2::signal<void()> embedded_post_refinemnt_signal;

    /**
     * Signals that the space grid has been just refined.
     */
    boost::signals2::signal<void()> space_post_refinemnt_signal;

  protected:
    /**
     * Embedded component mask.
     */
    const dealii::ComponentMask embedded_mask;

    /**
     * Space component mask.
     */
    const dealii::ComponentMask space_mask;

    /**
     * Embedded quadrature rule.
     */
    dealii::Quadrature<dim> embedded_quadrature;

    /**
     * Coupling type.
     */
    CouplingType coupling_type;

    /**
     * Refinement strategy.
     */
    RefinementStrategy refinement_strategy;

    /**
     * Pre refinement to apply on the space mesh.
     */
    unsigned int space_pre_refinement;

    /**
     * Post refinement to apply on the embedded mesh.
     */
    unsigned int embedded_post_refinement;

    /**
     * Embedded quadrature type.
     */
    std::string embedded_quadrature_type;

    /**
     * Order of the base embedded quadrature.
     */
    unsigned int quadrature_order;

    /**
     * Number of iterations of the base embedded quadrature.
     */
    unsigned int embedded_quadrature_repetitions;

    /**
     * Space cache.
     */
    dealii::SmartPointer<const dealii::GridTools::Cache<spacedim, spacedim>,
                         NonMatchingCoupling<dim, spacedim>>
      space_cache;

    /**
     * Space dof handler.
     */
    dealii::SmartPointer<const dealii::DoFHandler<spacedim, spacedim>,
                         NonMatchingCoupling<dim, spacedim>>
      space_dh;

    /**
     * Space constraints.
     */
    dealii::SmartPointer<const dealii::AffineConstraints<double>,
                         NonMatchingCoupling<dim, spacedim>>
      space_constraints;

    /**
     * Embedded cache.
     */
    dealii::SmartPointer<const dealii::GridTools::Cache<dim, spacedim>,
                         NonMatchingCoupling<dim, spacedim>>
      embedded_cache;

    /**
     * Embedded dof handler.
     */
    dealii::SmartPointer<const dealii::DoFHandler<dim, spacedim>,
                         NonMatchingCoupling<dim, spacedim>>
      embedded_dh;

    /**
     * Embedded constraints.
     */
    dealii::SmartPointer<const dealii::AffineConstraints<double>,
                         NonMatchingCoupling<dim, spacedim>>
      embedded_constraints;

    /**
     * Quadrature tolerance.
     *
     * If an intersection integrates to a value smaller than this tolerance, it
     * is discarded during exact intersection.
     */
    double quadrature_tolerance;

    /**
     * Store cache information.
     */
    mutable std::vector<std::tuple<
      typename dealii::Triangulation<spacedim, spacedim>::cell_iterator,
      typename dealii::Triangulation<dim, spacedim>::cell_iterator,
      dealii::Quadrature<spacedim>>>
      cells_and_quads;
  };


#ifndef DOXYGEN
  // Template instantiations
  template <int dim, int spacedim>
  template <typename MatrixType>
  void
  NonMatchingCoupling<dim, spacedim>::assemble_matrix(MatrixType &matrix) const
  {
    const auto &space_mapping    = space_cache->get_mapping();
    const auto &embedded_mapping = embedded_cache->get_mapping();
    if (coupling_type == CouplingType::approximate_L2)
      {
        dealii::NonMatching::create_coupling_mass_matrix(*space_cache,
                                                         *space_dh,
                                                         *embedded_dh,
                                                         embedded_quadrature,
                                                         matrix,
                                                         *space_constraints,
                                                         space_mask,
                                                         embedded_mask,
                                                         embedded_mapping,
                                                         *embedded_constraints);
      }
    else if (coupling_type == CouplingType::exact_L2)
      {
        if (cells_and_quads.empty() == true)
          cells_and_quads = dealii::NonMatching::compute_intersection(
            *space_cache,
            *embedded_cache,
            this->quadrature_order,
            this->quadrature_tolerance);

        dealii::NonMatching::
          assemble_coupling_mass_matrix_with_exact_intersections(
            *space_dh,
            *embedded_dh,
            cells_and_quads,
            matrix,
            *space_constraints,
            space_mask,
            embedded_mask,
            space_mapping,
            embedded_mapping,
            *embedded_constraints);
      }
  }



  template <int dim, int spacedim>
  template <typename SparsityType>
  void
  NonMatchingCoupling<dim, spacedim>::assemble_sparsity(SparsityType &dsp) const
  {
    Assert(space_dh, dealii::ExcNotInitialized());

    if (coupling_type == CouplingType::approximate_L2)
      {
        const auto &embedded_mapping = embedded_cache->get_mapping();

        dealii::NonMatching::create_coupling_sparsity_pattern(
          *space_cache,
          *space_dh,
          *embedded_dh,
          embedded_quadrature,
          dsp,
          *space_constraints,
          space_mask,
          embedded_mask,
          embedded_mapping,
          *embedded_constraints);
      }
    else if (coupling_type == CouplingType::exact_L2)
      {
        cells_and_quads.clear();
        cells_and_quads =
          dealii::NonMatching::compute_intersection(*space_cache,
                                                    *embedded_cache,
                                                    this->quadrature_order,
                                                    this->quadrature_tolerance);
        dealii::NonMatching::
          create_coupling_sparsity_pattern_with_exact_intersections(
            cells_and_quads,
            *space_dh,
            *embedded_dh,
            dsp,
            *space_constraints,
            space_mask,
            embedded_mask,
            *embedded_constraints);
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The requested coupling type is not implemented."));
      }
  }

#endif

} // namespace ParsedTools

#endif
