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

#include "pdes/linear_elasticity.h"

#include "deal.II/meshworker/mesh_loop.h"

#include "parsed_tools/components.h"

using namespace dealii;

namespace PDEs
{
  template <int dim, int spacedim, class LacType>
  LinearElasticity<dim, spacedim, LacType>::LinearElasticity()
    : LinearProblem<dim, spacedim, LacType>(
        ParsedTools::Components::join(std::vector<std::string>(spacedim, "u"),
                                      ","),
        "LinearElasticity")
    , lambda("/LinearElasticity/Lame coefficients", "1.0", "lambda")
    , mu("/LinearElasticity/Lame coefficients", "1.0", "mu")
    , displacement(0)
  {
    this->output_results_call_back.connect([&]() { postprocess(); });
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearElasticity<dim, spacedim, LacType>::assemble_system_one_cell(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    ScratchData &                                                   scratch,
    CopyData &                                                      copy)
  {
    auto &cell_matrix = copy.matrices[0];
    auto &cell_rhs    = copy.vectors[0];

    cell->get_dof_indices(copy.local_dof_indices[0]);

    const auto &fe_values = scratch.reinit(cell);
    cell_matrix           = 0;
    cell_rhs              = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
          {
            const auto  x = fe_values.quadrature_point(q_index);
            const auto &eps_v =
              fe_values[displacement].symmetric_gradient(i, q_index);
            const auto &div_v = fe_values[displacement].divergence(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                const auto &eps_u =
                  fe_values[displacement].symmetric_gradient(j, q_index);
                const auto &div_u =
                  fe_values[displacement].divergence(j, q_index);
                cell_matrix(i, j) += (2 * mu.value(x) * eps_v * eps_u +
                                      lambda.value(x) * div_v * div_u) *
                                     fe_values.JxW(q_index); // dx
              }

            cell_rhs(i) +=
              (fe_values.shape_value(i, q_index) * // phi_i(x_q)
               this->forcing_term.value(x,
                                        this->finite_element()
                                          .system_to_component_index(i)
                                          .first) * // f(x_q)
               fe_values.JxW(q_index));             // dx
          }
      }
  }



  template <int dim, int spacedim, class LacType>
  void
  LinearElasticity<dim, spacedim, LacType>::solve()
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
  LinearElasticity<dim, spacedim, LacType>::postprocess()
  {
    // Construct an object that will integrate on the faces only
    std::map<types::boundary_id, Tensor<1, spacedim>> forces;

    const auto face_integrator = [&](const auto &cell,
                                     const auto &face,
                                     auto &      scratch,
                                     auto &      data) {
      data[cell->face(face)->boundary_id()] = Tensor<1, spacedim>();

      auto &f = data[cell->face(face)->boundary_id()];

      const auto &fe_face_values = scratch.reinit(cell, face);
      scratch.extract_local_dof_values("solution",
                                       this->locally_relevant_solution);
      const auto &eps_u =
        scratch.get_symmetric_gradients("solution", displacement);

      const auto &div_u = scratch.get_divergences("solution", displacement);

      const auto &n   = scratch.get_normal_vectors();
      const auto &JxW = scratch.get_JxW_values();

      for (unsigned int q = 0; q < n.size(); ++q)
        f +=
          (2 * mu.value(fe_face_values.quadrature_point(q)) * eps_u[q] * n[q] +
           lambda.value(fe_face_values.quadrature_point(q)) * div_u[q] * n[q]) *
          JxW[q];
    };

    const auto copyer = [&](const auto &data) {
      for (const auto &[id, f] : data)
        {
          if (forces.find(id) == forces.end())
            forces[id] = f;
          else
            forces[id] += f;
        };
    };

    Quadrature<dim> quadrature_formula =
      ParsedTools::Components::get_cell_quadrature(
        this->triangulation, this->finite_element().tensor_degree() + 1);

    Quadrature<dim - 1> face_quadrature_formula =
      ParsedTools::Components::get_face_quadrature(
        this->triangulation, this->finite_element().tensor_degree() + 1);

    ScratchData scratch(*this->mapping,
                        this->finite_element(),
                        quadrature_formula,
                        update_default,
                        face_quadrature_formula,
                        update_values | update_gradients |
                          update_normal_vectors | update_quadrature_points |
                          update_JxW_values);


    using CellFilter = FilteredIterator<
      typename DoFHandler<dim, spacedim>::active_cell_iterator>;

    MeshWorker::mesh_loop(CellFilter(IteratorFilters::LocallyOwnedCell(),
                                     this->dof_handler.begin_active()),
                          CellFilter(IteratorFilters::LocallyOwnedCell(),
                                     this->dof_handler.end()),
                          {},
                          copyer,
                          scratch,
                          forces,
                          MeshWorker::assemble_boundary_faces,
                          face_integrator);

    this->pcout << "Forces: " << std::endl;
    for (const auto &[id, force] : forces)
      this->pcout << "ID " << id << ": " << force << std::endl;
  }



  template class LinearElasticity<2, 2, LAC::LAdealii>;
  template class LinearElasticity<2, 3, LAC::LAdealii>;
  template class LinearElasticity<3, 3, LAC::LAdealii>;

  template class LinearElasticity<2, 2, LAC::LATrilinos>;
  template class LinearElasticity<2, 3, LAC::LATrilinos>;
  template class LinearElasticity<3, 3, LAC::LATrilinos>;
} // namespace PDEs