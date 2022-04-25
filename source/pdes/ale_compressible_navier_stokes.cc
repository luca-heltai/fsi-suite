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
#include "pdes/ale_compressible_navier_stokes.h"

#include <deal.II/base/symmetric_tensor.h>

#include "deal.II/meshworker/mesh_loop.h"

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, int spacedim, class LacType>
  ALECompressibleNavierStokes<dim, spacedim, LacType>::
    ALECompressibleNavierStokes()
    : LinearProblem<dim, spacedim, LacType>(
        ParsedTools::Components::blocks_to_names({"w", "u", "rho"},
                                                 {spacedim, spacedim, 1}),
        "ALECompressibleNavierStokes")
    , displacement(0)
    , velocity(spacedim)
    , rho(2 * spacedim)
    , constants_0("/ALECompressibleNavierStokes/Constants 0",
                  {"mu", "lambda", "eta", "lambda_viscous", "kappa"},
                  {1.0, 1.0, 0.0, 0.0, 0.0},
                  {"First Lame coefficient",
                   "Second Lame coefficient",
                   "Shear viscosity",
                   "Bulk lame' viscosity",
                   "Bulk viscosity"})
    , constants_1("/ALECompressibleNavierStokes/Constants 1",
                  {"mu", "lambda", "eta", "lambda_viscous", "kappa"},
                  {0.0, 0.0, 1.0, 1.0, 1.0},
                  {"First Lame coefficient",
                   "Second Lame coefficient",
                   "Shear viscosity",
                   "Bulk lame' viscosity",
                   "Bulk viscosity"})
    , material_ids_0({{0}})
    , eulerian_mapping(this->dof_handler,
                       "/ALECompressibleNavierStokes/Mapping",
                       "",
                       true)
  //    // Select first spacedim compoments for the displacement
  //    [&]() {
  //      std::vector<bool> mask(2 * spacedim + 1, false);
  //      for (unsigned int i = 0; i < spacedim; ++i)
  //        mask[i] = true;
  //      return mask;
  //    }())
  {
    this->add_parameter("Material ids of region 0", material_ids_0);
    this->add_parameter("Material ids of region 1", material_ids_1);

    this->output_results_call_back.connect([&]() { postprocess(); });
    this->check_consistency_call_back.connect([&]() {
      AssertThrow(
        this->evolution_type != EvolutionType::transient,
        ExcMessage(
          "This code won't produce correct results in transient simulations. "
          "Run the wave_equation code instead."));
      //   const auto m = this->triangulation().get_material_ids();

      auto all_m = material_ids_0;
      all_m.insert(material_ids_1.begin(), material_ids_1.end());

      AssertThrow(all_m.size() ==
                    (material_ids_0.size() + material_ids_1.size()),
                  ExcMessage("You cannot assign the same material id to two "
                             "different regions"));
    });

    this->advance_time_call_back.connect(
      [&](const auto &t, const auto &dt, const auto &n) {
        this->current_time  = t;
        this->dt            = dt;
        this->current_cycle = n;
      });
  }



  template <int dim, int spacedim, class LacType>
  void
  ALECompressibleNavierStokes<dim, spacedim, LacType>::assemble_system()
  {
    TimerOutput::Scope timer_section(this->timer, "assemble_system");
    Quadrature<dim>    quadrature_formula =
      ParsedTools::Components::get_cell_quadrature(
        this->triangulation, this->finite_element().tensor_degree() + 1);

    // Initialize the mapping, if necessary.
    if (current_cycle == 0)
      eulerian_mapping.initialize(this->solution,
                                  this->locally_relevant_solution);

    ScratchData scratch(this->finite_element(),
                        quadrature_formula,
                        update_gradients | update_JxW_values);

    ScratchData mapped_scratch(eulerian_mapping(),
                               this->finite_element(),
                               quadrature_formula,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);

    CopyData copy(this->finite_element().n_dofs_per_cell());

    this->rhs    = 0;
    this->matrix = 0;

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          auto &cell_matrix = copy.matrices[0];
          auto &cell_rhs    = copy.vectors[0];

          cell->get_dof_indices(copy.local_dof_indices[0]);

          const auto &fe_values        = scratch.reinit(cell);
          const auto &mapped_fe_values = mapped_scratch.reinit(cell);

          cell_matrix = 0;
          cell_rhs    = 0;

          // The displacement field part.
          scratch.extract_local_dof_values("sol",
                                           this->locally_relevant_solution);
          mapped_scratch.extract_local_dof_values(
            "sol", this->locally_relevant_solution);

          //   const auto &div_Wn = scratch.get_divergences("sol",
          //   displacement); const auto &eps_Wn =
          //     scratch.get_symmetric_gradients("sol", displacement);

          // The current rho, grad_rho, and grad_u parts
          const auto &rho_n      = mapped_scratch.get_values("sol", rho);
          const auto &grad_rho_n = mapped_scratch.get_gradients("sol", rho);
          const auto &grad_u_n = mapped_scratch.get_gradients("sol", velocity);
          const auto &u_n      = mapped_scratch.get_values("sol", velocity);

          const ParsedTools::Constants &c =
            (material_ids_0.find(cell->material_id()) != material_ids_0.end()) ?
              constants_0 :
              constants_1;

          static const auto identity = unit_symmetric_tensor<spacedim>();

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              {
                const auto x = mapped_fe_values.quadrature_point(q_index);

                // Lagrangian
                const auto &Grad_V = fe_values[velocity].gradient(i, q_index);

                // const auto &eps_Wtest =
                //   fe_values[displacement].symmetric_gradient(i, q_index);

                // Eulerian test function for velocity
                const auto &v = mapped_fe_values[velocity].value(i, q_index);

                const auto &eps_v =
                  mapped_fe_values[velocity].symmetric_gradient(i, q_index);

                // Eulerian test function for rho
                const auto &q = mapped_fe_values[rho].value(i, q_index);

                for (const unsigned int j : fe_values.dof_indices())
                  {
                    // Elastic part.
                    const auto &eps_W =
                      fe_values[displacement].symmetric_gradient(j, q_index);
                    const auto &div_W =
                      fe_values[displacement].divergence(j, q_index);

                    const auto P_el =
                      2 * c["mu"] * eps_W + c["lambda"] * div_W * identity;

                    // Fluid part
                    const auto &u =
                      mapped_fe_values[velocity].value(j, q_index);

                    const auto &eps_u =
                      mapped_fe_values[velocity].symmetric_gradient(j, q_index);

                    const auto &div_u =
                      mapped_fe_values[velocity].divergence(j, q_index);

                    const auto &rho =
                      mapped_fe_values[this->rho].value(j, q_index);

                    const auto sigma_vis =
                      2 * c["eta"] * eps_u +
                      c["lambda_viscous"] * div_u * identity +
                      c["kappa"] * rho * identity;

                    cell_matrix(i, j) +=
                      // (rho_n  u_dot, v)
                      (rho_n[q_index] * u * v / dt +
                       // (sigma, eps(v))
                       scalar_product(sigma_vis, eps_v) +
                       // (rho_dot + rho div u, q)
                       (rho / dt + rho_n[q_index] * div_u) * q) *
                        mapped_fe_values.JxW(q_index) // dx
                      +
                      // P:Grad(V)
                      scalar_product(P_el, Grad_V) *
                        fe_values.JxW(q_index); // dX
                  }

                cell_rhs(i) +=
                  (rho_n[q_index] *
                     (u_n[q_index] / dt - grad_u_n[q_index] * u_n[q_index]) *
                     v +
                   (rho_n[q_index] / dt - grad_rho_n[q_index] * u_n[q_index]) *
                     q +
                   mapped_fe_values.shape_value(i, q_index) * // phi_i(x_q)
                     this->forcing_term.value(x,
                                              this->finite_element()
                                                .system_to_component_index(i)
                                                .first) * // f(x_q)
                     fe_values.JxW(q_index));             // dx
              }
          this->constraints.distribute_local_to_global(
            cell_matrix,
            cell_rhs,
            copy.local_dof_indices[0],
            this->matrix,
            this->rhs);
        }
    this->matrix.compress(VectorOperation::add);
    this->rhs.compress(VectorOperation::add);
  }



  template <int dim, int spacedim, class LacType>
  void
  ALECompressibleNavierStokes<dim, spacedim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(this->timer, "solve");
    const auto A = linear_operator<VectorType>(this->matrix.block(0, 0));
    this->preconditioner.initialize(this->matrix.block(0, 0));
    const auto Ainv         = this->inverse_operator(A, this->preconditioner);
    this->solution.block(0) = Ainv * this->rhs.block(0);
    this->constraints.distribute(this->solution);
    this->locally_relevant_solution = this->solution;
  }



  template <int dim, int spacedim, class LacType>
  void
  ALECompressibleNavierStokes<dim, spacedim, LacType>::postprocess()
  {
    // TimerOutput::Scope timer_section(this->timer, "post_process");
    // Quadrature<dim>    quadrature_formula =
    //   ParsedTools::Components::get_cell_quadrature(
    //     this->triangulation, this->finite_element().tensor_degree() + 1);

    // ScratchData scratch(this->finite_element(),
    //                     quadrature_formula,
    //                     update_gradients | update_JxW_values);

    // ScratchData mapped_scratch(eulerian_mapping(),
    //                            this->finite_element(),
    //                            quadrature_formula,
    //                            update_quadrature_points | update_values |
    //                              update_gradients | update_JxW_values);

    // static std::ofstream outfile("visco_elastic_energies.txt");
    // if (current_cycle == 0)
    //   outfile
    //     << "# t \t E_pot[0] \t E_pot[1] \t Diss[0] \t Diss[1] \t Ext[0] \t
    //     Ext[1]"
    //     << std::endl;

    // // One per material
    // std::vector<double> potential_energy(2, 0.0);
    // std::vector<double> dissipation(2, 0.0);
    // std::vector<double> external_energy(2, 0.0);

    // for (const auto &cell : this->dof_handler.active_cell_iterators())
    //   if (cell->is_locally_owned())
    //     {
    //       const auto &fe_values        = scratch.reinit(cell);
    //       const auto &mapped_fe_values = mapped_scratch.reinit(cell);

    //       scratch.extract_local_dof_values(
    //         "Wn", this->current_displacement_locally_relevant);
    //       mapped_scratch.extract_local_dof_values(
    //         "Un", this->locally_relevant_solution);

    //       const auto &div_Wn = scratch.get_divergences("Wn", displacement);
    //       const auto &eps_Wn =
    //         scratch.get_symmetric_gradients("Wn", displacement);


    //       const auto &un = mapped_scratch.get_values("Un", displacement);
    //       const auto &div_un =
    //         mapped_scratch.get_divergences("Un", displacement);
    //       const auto &eps_un =
    //         mapped_scratch.get_symmetric_gradients("Un", displacement);

    //       const auto id =
    //         material_ids_0.find(cell->material_id()) != material_ids_0.end()
    //         ?
    //           0 :
    //           1;

    //       const ParsedTools::Constants &c =
    //         (id == 0 ? constants_0 : constants_1);

    //       for (const unsigned int q_index :
    //            fe_values.quadrature_point_indices())
    //         {
    //           potential_energy[id] +=
    //             c["mu"] * scalar_product(eps_Wn[q_index], eps_Wn[q_index]) +
    //             0.5 * c["lambda"] * div_Wn[q_index] * div_Wn[q_index] *
    //               fe_values.JxW(q_index);

    //           dissipation[id] +=
    //             c["eta"] * scalar_product(eps_un[q_index], eps_un[q_index]) +
    //             0.5 * c["kappa"] * div_un[q_index] * div_un[q_index] *
    //               mapped_fe_values.JxW(q_index);

    //           for (unsigned int i = 0; i < spacedim; ++i)
    //             external_energy[id] +=
    //               un[q_index][i] *
    //               this->forcing_term.value(
    //                 mapped_fe_values.quadrature_point(q_index), i) *
    //               mapped_fe_values.JxW(q_index);
    //         }
    //     }

    // // Sum over all processors
    // Utilities::MPI::sum(ArrayView<const double>(potential_energy),
    //                     this->mpi_communicator,
    //                     ArrayView<double>(potential_energy));
    // Utilities::MPI::sum(ArrayView<const double>(dissipation),
    //                     this->mpi_communicator,
    //                     ArrayView<double>(dissipation));
    // Utilities::MPI::sum(ArrayView<const double>(external_energy),
    //                     this->mpi_communicator,
    //                     ArrayView<double>(external_energy));

    // outfile << current_time << " \t " << potential_energy[0] << " \t "
    //         << potential_energy[1] << " \t " << dissipation[0] << " \t "
    //         << dissipation[1] << " \t "
    //         << external_energy[0] + external_energy[1] << std::endl;
  }



  template class ALECompressibleNavierStokes<2, 2, LAC::LAdealii>;
  template class ALECompressibleNavierStokes<2, 3, LAC::LAdealii>;
  template class ALECompressibleNavierStokes<3, 3, LAC::LAdealii>;

  template class ALECompressibleNavierStokes<2, 2, LAC::LATrilinos>;
  template class ALECompressibleNavierStokes<2, 3, LAC::LATrilinos>;
  template class ALECompressibleNavierStokes<3, 3, LAC::LATrilinos>;
} // namespace PDEs