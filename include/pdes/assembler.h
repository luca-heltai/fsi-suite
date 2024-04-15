// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by Luca Heltai
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

// Make sure we don't redefine things
#ifndef pdes_assembler_h
#define pdes_assembler_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <Sacado.hpp>

namespace PDEs
{
  using namespace dealii;

  /**
   * Provides an assembler for a generic Matrix.
   */
  template <class LocalAssembler, typename NumberType = double>
  class Assembler : public LocalAssembler
  {
  public:
    using SacadoNumber       = Sacado::Fad::DFad<NumberType>;
    using SacadoSacadoNumber = Sacado::Fad::DFad<SacadoNumber>;

    Assembler() = default;

    template <typename MatrixType, typename DoFHandlerType>
    void
    assemble(MatrixType &matrix, const DoFHandlerType &dof_handler)
    {
      auto scratch = this->get_scratch();
      auto copy    = this->get_copy();

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            assemble_local(cell, scratch, copy);
            copy_local_to_global(copy, matrix);
          }
      matrix.compress(VectorOperation::add);
    }


    void
    assemble_energies_and_residuals(typename LocalAssembler::CellType &cell,
                                    typename LocalAssembler::Scratch  &scratch,
                                    typename LocalAssembler::Copy     &copy,
                                    SacadoNumber                      &energies,
                                    std::vector<NumberType> &residual) const
    {
      static_cast<const LocalAssembler *>(this)->energies_and_residuals(
        cell, scratch, energies, local_residuals);
    }

    void
    assemble_local(typename LocalAssembler::Scratch &scratch,
                   typename LocalAssembler::Copy    &copy)
    {
      this->local_assemble(scratch, copy);
    }
  }

};
} // namespace PDEs
#endif