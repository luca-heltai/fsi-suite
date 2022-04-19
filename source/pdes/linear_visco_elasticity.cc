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
#include "pdes/linear_visco_elasticity.h"

#include <deal.II/base/symmetric_tensor.h>

#include "deal.II/meshworker/mesh_loop.h"

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, int spacedim, class LacType>
  LinearViscoElasticity<dim, spacedim, LacType>::LinearViscoElasticity()
    : LinearProblem<dim, spacedim, LacType>(
        ParsedTools::Components::blocks_to_names({"U"}, {spacedim}),
        "LinearViscoElasticity")
    , displacement(0)
    , constants_0("/LinearViscoElasticity/Constants 0",
                  {"mu", "lambda", "eta", "kappa"},
                  {1.0, 1.0, 0.0, 0.0},
                  {"First Lame coefficient",
                   "Second Lame coefficient",
                   "Shear viscosity",
                   "Bulk viscosity"})
    , constants_1("/LinearViscoElasticity/Constants 1",
                  {"mu", "lambda", "eta", "kappa"},
                  {0.0, 0.0, 1.0, 1.0},
                  {"First Lame coefficient",
                   "Second Lame coefficient",
                   "Shear viscosity",
                   "Bulk viscosity"})
    , material_ids_0({{0}})
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

      //   AssertThrow(m.size() == (material_ids_0.size() +
      //   material_ids_1.size()),
      //               ExcMessage(
      //                 "The mesh must have the same number of material ids "
      //                 "as the number of material ids you specified "
      //                 "in the parameter file"));
    });

    this->setup_system_call_back.connect([&]() {
      // Make sure we only setup the displacement vector once
      if (!eulerian_mapping)
        {
          current_displacement_locally_relevant.reinit(
            this->locally_relevant_solution);
          current_displacement.reinit(this->solution);

          eulerian_mapping = std::make_unique<
            MappingQEulerian<dim, typename LacType::BlockVector, spacedim>>(
            this->finite_element().degree,
            this->dof_handler,
            current_displacement_locally_relevant);
        }
    });

    this->advance_time_call_back.connect(
      [&](const auto &t, const auto &dt, const auto &n) {
        this->current_time  = t;
        this->dt            = dt;
        this->current_cycle = n;
      });

    this->add_data_vector.connect([&](auto &d) {
      d.add_data_vector(current_displacement_locally_relevant,
                        ParsedTools::Components::blocks_to_names({"W"},
                                                                 {spacedim}),
                        dealii::DataOut<dim, spacedim>::type_dof_data);
    });
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearViscoElasticity<dim, spacedim, LacType>::assemble_system()
  {
    TimerOutput::Scope timer_section(this->timer, "assemble_system");
    Quadrature<dim>    quadrature_formula =
      ParsedTools::Components::get_cell_quadrature(
        this->triangulation, this->finite_element().tensor_degree() + 1);


    ScratchData scratch(this->finite_element(),
                        quadrature_formula,
                        update_gradients | update_JxW_values);

    ScratchData mapped_scratch(*eulerian_mapping,
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

          scratch.extract_local_dof_values(
            "Wn", this->current_displacement_locally_relevant);
          const auto &div_Wn = scratch.get_divergences("Wn", displacement);
          const auto &eps_Wn =
            scratch.get_symmetric_gradients("Wn", displacement);

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
                const auto &eps_V =
                  fe_values[displacement].symmetric_gradient(i, q_index);

                // Eulerian
                const auto eps_v =
                  mapped_fe_values[displacement].symmetric_gradient(i, q_index);

                for (const unsigned int j : fe_values.dof_indices())
                  {
                    // Elastic part.
                    const auto &eps_W =
                      fe_values[displacement].symmetric_gradient(j, q_index);
                    const auto &div_W =
                      fe_values[displacement].divergence(j, q_index);

                    const auto P_el = this->dt * 2 * c["mu"] * eps_W +
                                      c["lambda"] * div_W * identity;

                    const auto &eps_u =
                      mapped_fe_values[displacement].symmetric_gradient(
                        j, q_index);

                    const auto &div_u =
                      mapped_fe_values[displacement].divergence(j, q_index);

                    const auto sigma_vis =
                      2 * c["eta"] * eps_u + c["kappa"] * div_u * identity;

                    cell_matrix(i, j) +=
                      (scalar_product(sigma_vis, eps_v) *
                         mapped_fe_values.JxW(q_index) + // dx
                       scalar_product(P_el, eps_V) *
                         fe_values.JxW(q_index)); // dX
                  }

                const auto Pn = 2 * c["mu"] * eps_Wn[q_index] +
                                c["lambda"] * div_Wn[q_index] * identity;

                cell_rhs(i) +=
                  (-scalar_product(Pn, eps_V) * fe_values.JxW(q_index) + // dX
                   mapped_fe_values.shape_value(i, q_index) * // phi_i(x_q)
                     this->forcing_term.value(x,
                                              this->finite_element()
                                                .system_to_component_index(i)
                                                .first) * // f(x_q)
                     mapped_fe_values.JxW(q_index));      // dx
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
  LinearViscoElasticity<dim, spacedim, LacType>::solve()
  {
    TimerOutput::Scope timer_section(this->timer, "solve");
    const auto A = linear_operator<VectorType>(this->matrix.block(0, 0));
    this->preconditioner.initialize(this->matrix.block(0, 0));
    const auto Ainv         = this->inverse_operator(A, this->preconditioner);
    this->solution.block(0) = Ainv * this->rhs.block(0);
    this->constraints.distribute(this->solution);
    this->locally_relevant_solution = this->solution;
    current_displacement.sadd(1.0, dt, this->solution);
    current_displacement_locally_relevant = current_displacement;
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearViscoElasticity<dim, spacedim, LacType>::postprocess()
  {
    TimerOutput::Scope timer_section(this->timer, "post_process");
    Quadrature<dim>    quadrature_formula =
      ParsedTools::Components::get_cell_quadrature(
        this->triangulation, this->finite_element().tensor_degree() + 1);

    ScratchData scratch(this->finite_element(),
                        quadrature_formula,
                        update_gradients | update_JxW_values);

    ScratchData mapped_scratch(*eulerian_mapping,
                               this->finite_element(),
                               quadrature_formula,
                               update_quadrature_points | update_values |
                                 update_gradients | update_JxW_values);

    static std::ofstream outfile("visco_elastic_energies.txt");
    if (current_cycle == 0)
      outfile
        << "# t \t E_pot[0] \t E_pot[1] \t Diss[0] \t Diss[1] \t Ext[0] \t Ext[1]"
        << std::endl;

    // One per material
    std::vector<double> potential_energy(2, 0.0);
    std::vector<double> dissipation(2, 0.0);
    std::vector<double> external_energy(2, 0.0);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &fe_values        = scratch.reinit(cell);
          const auto &mapped_fe_values = mapped_scratch.reinit(cell);

          scratch.extract_local_dof_values(
            "Wn", this->current_displacement_locally_relevant);
          mapped_scratch.extract_local_dof_values(
            "Un", this->locally_relevant_solution);

          const auto &div_Wn = scratch.get_divergences("Wn", displacement);
          const auto &eps_Wn =
            scratch.get_symmetric_gradients("Wn", displacement);


          const auto &un = mapped_scratch.get_values("Un", displacement);
          const auto &div_un =
            mapped_scratch.get_divergences("Un", displacement);
          const auto &eps_un =
            mapped_scratch.get_symmetric_gradients("Un", displacement);

          const auto id =
            material_ids_0.find(cell->material_id()) != material_ids_0.end() ?
              0 :
              1;

          const ParsedTools::Constants &c =
            (id == 0 ? constants_0 : constants_1);

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              potential_energy[id] +=
                c["mu"] * scalar_product(eps_Wn[q_index], eps_Wn[q_index]) +
                0.5 * c["lambda"] * div_Wn[q_index] * div_Wn[q_index] *
                  fe_values.JxW(q_index);

              dissipation[id] +=
                c["eta"] * scalar_product(eps_un[q_index], eps_un[q_index]) +
                0.5 * c["kappa"] * div_un[q_index] * div_un[q_index] *
                  mapped_fe_values.JxW(q_index);

              for (unsigned int i = 0; i < spacedim; ++i)
                external_energy[id] +=
                  un[q_index][i] *
                  this->forcing_term.value(
                    mapped_fe_values.quadrature_point(q_index), i) *
                  mapped_fe_values.JxW(q_index);
            }
        }

    // Sum over all processors
    Utilities::MPI::sum(ArrayView<const double>(potential_energy),
                        this->mpi_communicator,
                        ArrayView<double>(potential_energy));
    Utilities::MPI::sum(ArrayView<const double>(dissipation),
                        this->mpi_communicator,
                        ArrayView<double>(dissipation));
    Utilities::MPI::sum(ArrayView<const double>(external_energy),
                        this->mpi_communicator,
                        ArrayView<double>(external_energy));

    outfile << current_time << " \t " << potential_energy[0] << " \t "
            << potential_energy[1] << " \t " << dissipation[0] << " \t "
            << dissipation[1] << " \t "
            << external_energy[0] + external_energy[1] << std::endl;
  }



  template class LinearViscoElasticity<2, 2, LAC::LAdealii>;
  template class LinearViscoElasticity<2, 3, LAC::LAdealii>;
  template class LinearViscoElasticity<3, 3, LAC::LAdealii>;

  template class LinearViscoElasticity<2, 2, LAC::LATrilinos>;
  template class LinearViscoElasticity<2, 3, LAC::LATrilinos>;
  template class LinearViscoElasticity<3, 3, LAC::LATrilinos>;
} // namespace PDEs