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
#ifndef pdes_linear_visco_elasticity_h
#define pdes_linear_visco_elasticity_h

#include "parsed_tools/constants.h"
#include "parsed_tools/mapping_eulerian.h"
#include "pdes/linear_problem.h"

namespace PDEs
{
  using namespace dealii;

  /**
   * Serial LinearViscoElasticity problem.
   *
   * @addtogroup csd
   * @addtogroup cfd
   *
   * This tutorial program shows how to model a linearily viscous elastic
   * material made of two phases. This is a prototype for general elastic
   * materials, as well as for compressible fluid flow problems.
   *
   * We assume that the domain is split into two regions, identified by
   * differents sets of material ids, and that the material properties of the
   * two regions maybe different.
   *
   * In particular we can specify, for each material, the two Lame parameters,
   * the shear viscosity and the bulk viscosity.
   */
  template <int dim, int spacedim = dim, class LacType = LAC::LAdealii>
  class LinearViscoElasticity : public LinearProblem<dim, spacedim, LacType>
  {
  public:
    /**
     * Constructor. Initialize all parameters, including the base class, and
     * make sure the class is ready to run.
     */
    LinearViscoElasticity();

    /**
     * Destroy the LinearElasticity object
     */
    virtual ~LinearViscoElasticity() = default;

    using ScratchData =
      typename LinearProblem<dim, spacedim, LacType>::ScratchData;

    using CopyData = typename LinearProblem<dim, spacedim, LacType>::CopyData;

    using VectorType =
      typename LinearProblem<dim, spacedim, LacType>::VectorType;

    /**
     * Compute energy integrals.
     */
    void
    postprocess();

  protected:
    /**
     * Explicitly assemble the LinearViscoElasticity problem.
     */
    virtual void
    assemble_system() override;

    /**
     * Make sure we initialize the right type of linear solver.
     */
    virtual void
    solve() override;

    /**
     * Displacement field.
     */
    const FEValuesExtractors::Vector displacement;

    /**
     * Constants of the first material.
     */
    ParsedTools::Constants constants_0;

    /**
     * Constants of the second material.
     */
    ParsedTools::Constants constants_1;

    /**
     * Material ids of the first material.
     */
    std::set<types::material_id> material_ids_0;

    /**
     * Material ids of the second material.
     */
    std::set<types::material_id> material_ids_1;

    /**
     * Mapping from reference configuration to deformed configuration.
     */
    std::unique_ptr<
      dealii::MappingQEulerian<dim, typename LacType::BlockVector, spacedim>>
      eulerian_mapping;

    /**
     * Time step.
     */
    double dt;

    /**
     * Current time.
     */
    double current_time;

    /**
     * Current cycle.
     */
    unsigned int current_cycle;

    typename LacType::BlockVector current_displacement;
    typename LacType::BlockVector current_displacement_locally_relevant;
  };

  namespace MPI
  {
    template <int dim, int spacedim = dim>
    using LinearViscoElasticity =
      PDEs::LinearViscoElasticity<dim, spacedim, LAC::LATrilinos>;
  }

  namespace Serial
  {
    template <int dim, int spacedim = dim>
    using LinearViscoElasticity =
      PDEs::LinearViscoElasticity<dim, spacedim, LAC::LAdealii>;
  }
} // namespace PDEs
#endif