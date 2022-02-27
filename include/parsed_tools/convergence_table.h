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

#ifndef parsed_tools_convergence_table_h
#define parsed_tools_convergence_table_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>

#include "parsed_tools/proxy.h"

namespace ParsedTools
{
  /**
   * A class that provides a wrapper around deal.II's
   * dealii::ParsedConvergenceTable class, and makes it a
   * dealii::ParameterAcceptor class as well.
   *
   * @copydetails dealii::ParsedConvergenceTable
   */
  using ConvergenceTable = Proxy<dealii::ParsedConvergenceTable>;
} // namespace ParsedTools
#endif