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
#ifndef pdes_ale_compressible_navier_stokes_h
#define pdes_ale_compressible_navier_stokes_h

#include "parsed_tools/constants.h"
#include "parsed_tools/mapping_eulerian.h"
#include "pdes/linear_problem.h"

namespace PDEs
{
  using namespace dealii;

  /**
   * ALE compressible Navier-Stokes problem.
   *
   * @addtogroup cfd
   */
  template <int dim, int spacedim = dim, class LacType = LAC::LAdealii>
  class ALECompressibleNavierStokes
    : public LinearProblem<dim, spacedim, LacType>
  {
  public:
    /**
     * Constructor. Initialize all parameters, including the base class, and
     * make sure the class is ready to run.
     */
    ALECompressibleNavierStokes();

    /**
     * Destroy the LinearElasticity object
     */
    virtual ~ALECompressibleNavierStokes() = default;

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
     * Explicitly assemble the ALECompressibleNavierStokes problem.
     */
    virtual void
    assemble_system() override;

    /**
     * Make sure we initialize the right type of linear solver.
     */
    virtual void
    solve() override;

    /**
     * Displacement field extractor.
     */
    const FEValuesExtractors::Vector displacement;

    /**
     * Velocity field extractor.
     */
    const FEValuesExtractors::Vector velocity;

    /**
     * Rho field extractor.
     */
    const FEValuesExtractors::Scalar rho;

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
    ParsedTools::MappingEulerian<dim, spacedim> eulerian_mapping;

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
  };

  namespace MPI
  {
    template <int dim, int spacedim = dim>
    using ALECompressibleNavierStokes =
      PDEs::ALECompressibleNavierStokes<dim, spacedim, LAC::LATrilinos>;
  }

  //   namespace MPI
  //   {
  //     template <int dim, int spacedim = dim>
  //     using ALECompressibleNavierStokes =
  //       PDEs::ALECompressibleNavierStokes<dim, spacedim, LAC::LAPETSc>;
  //   }

  namespace Serial
  {
    template <int dim, int spacedim = dim>
    using ALECompressibleNavierStokes =
      PDEs::ALECompressibleNavierStokes<dim, spacedim, LAC::LAdealii>;
  }
} // namespace PDEs
#endif