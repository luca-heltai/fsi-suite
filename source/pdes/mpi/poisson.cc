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

#include "pdes/mpi/poisson.h"

#include <deal.II/meshworker/mesh_loop.h>

using namespace dealii;

namespace PDEs
{
  namespace MPI
  {
    template <int dim, int spacedim>
    Poisson<dim, spacedim>::Poisson()
      : LinearProblem<dim, spacedim, LAC::LATrilinos>("u", "Poisson")
      , coefficient("/Poisson/Functions", "1", "Diffusion coefficient")
    {}



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::assemble_system_one_cell(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      ScratchData                                                    &scratch,
      CopyData                                                       &copy)
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
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (coefficient.value(
                   fe_values.quadrature_point(q_index)) * // a(x_q)
                 fe_values.shape_grad(i, q_index) *       // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) *       // grad phi_j(x_q)
                 fe_values.JxW(q_index));                 // dx
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            this->forcing_term.value(
                              fe_values.quadrature_point(q_index)) * // f(x_q)
                            fe_values.JxW(q_index));                 // dx
        }
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::solve()
    {
      TimerOutput::Scope timer_section(this->timer, "solve");
      const auto A = linear_operator<VectorType>(this->matrix.block(0, 0));
      this->preconditioner.initialize(this->matrix.block(0, 0));
      const auto Ainv         = this->inverse_operator(A, this->preconditioner);
      this->solution.block(0) = Ainv * this->rhs.block(0);
      this->constraints.distribute(this->solution);
      this->locally_relevant_solution = this->solution;
    }



    template <int dim, int spacedim>
    void
    Poisson<dim, spacedim>::custom_estimator(
      dealii::Vector<float> &error_per_cell) const
    {
      TimerOutput::Scope timer_section(this->timer, "custom_estimator");
      error_per_cell = 0;
      Quadrature<dim> quadrature_formula =
        ParsedTools::Components::get_cell_quadrature(
          this->triangulation, this->finite_element().tensor_degree() + 1);

      Quadrature<dim - 1> face_quadrature_formula =
        ParsedTools::Components::get_face_quadrature(
          this->triangulation, this->finite_element().tensor_degree() + 1);

      ScratchData scratch(*this->mapping,
                          this->finite_element(),
                          quadrature_formula,
                          update_quadrature_points | update_hessians |
                            update_JxW_values,
                          face_quadrature_formula,
                          update_normal_vectors | update_gradients |
                            update_quadrature_points | update_JxW_values);

      // A copy data for error estimators for each cell. We store the indices of
      // the cells, and the values of the error estimator to be added to the
      // cell indicators.
      struct MyCopyData
      {
        std::vector<unsigned int> cell_indices;
        std::vector<float>        indicators;
      };

      MyCopyData copy;

      // I will use this FEValuesExtractor to leverage the capabilities of the
      // ScratchData
      FEValuesExtractors::Scalar scalar(0);

      // This is called in each cell
      auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {
        const auto &fe_v = scratch.reinit(cell);
        const auto  H    = cell->diameter();

        // Reset the copy data
        copy.cell_indices.resize(0);
        copy.indicators.resize(0);

        // Save the index of this cell
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // At every call of this function, a new vector of dof values is
        // generated and stored internally, so that you can later call
        // scratch.get_values(...)
        scratch.extract_local_dof_values("solution",
                                         this->locally_relevant_solution);

        // Get the values of the solution at the quadrature points
        const auto &lap_u = scratch.get_laplacians("solution", scalar);

        // Points and weights of the quadrature formula
        const auto &q_points = scratch.get_quadrature_points();
        const auto &JxW      = scratch.get_JxW_values();

        // Reset vectors
        float cell_indicator = 0;

        // Now store the values of the residual square in the copy data
        for (const auto q_index : fe_v.quadrature_point_indices())
          {
            const auto res =
              lap_u[q_index] + this->forcing_term.value(q_points[q_index]);

            cell_indicator += (H * H * res * res * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(cell_indicator);
      };

      // This is called in each face, refined or not.
      auto face_worker = [&](const auto &cell,
                             const auto &f,
                             const auto &sf,
                             const auto &ncell,
                             const auto &nf,
                             const auto &nsf,
                             auto       &scratch,
                             auto       &copy) {
        // Here we intialize the inteface values
        const auto &fe_iv = scratch.reinit(cell, f, sf, ncell, nf, nsf);

        const auto h = cell->face(f)->diameter();

        // Add this cell to the copy data
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // Same as before. Extract local dof values of the solution
        scratch.extract_local_dof_values("solution",
                                         this->locally_relevant_solution);

        // ...so that we can call scratch.get_(...)
        const auto jump_grad =
          scratch.get_jumps_in_gradients("solution", scalar);

        const auto &JxW     = scratch.get_JxW_values();
        const auto &normals = scratch.get_normal_vectors();

        // Now store the values of the gradient jump in the copy data
        float face_indicator = 0;
        for (const auto q_index : fe_iv.quadrature_point_indices())
          {
            const auto J = jump_grad[q_index] * normals[q_index];

            face_indicator += (h * J * J * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(face_indicator);
      };


      auto copier = [&](const auto &copy) {
        AssertDimension(copy.cell_indices.size(), copy.indicators.size());
        for (unsigned int i = 0; i < copy.cell_indices.size(); ++i)
          {
            error_per_cell[copy.cell_indices[i]] += copy.indicators[i];
          }
      };

      using CellFilter = FilteredIterator<
        typename DoFHandler<dim, spacedim>::active_cell_iterator>;

      MeshWorker::mesh_loop(this->dof_handler.begin_active(),
                            this->dof_handler.end(),
                            cell_worker,
                            copier,
                            scratch,
                            copy,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_own_interior_faces_both,
                            {},
                            face_worker);

      // Collect errors from the other processors
      const double total_error =
        Utilities::MPI::sum(this->error_per_cell.l1_norm(),
                            this->mpi_communicator);

      deallog << "Error estimator: " << total_error << std::endl;
    }


    template class Poisson<2, 2>;
    template class Poisson<2, 3>;
    template class Poisson<3, 3>;
  } // namespace MPI
} // namespace PDEs