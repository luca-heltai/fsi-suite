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

#ifndef parsed_tools_proxy_h
#define parsed_tools_proxy_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>

namespace ParsedTools
{
  /**
   * A proxy ParameterAcceptor wrapper for classes that have a member
   * function @p add_parameters, which takes a ParameterHandler as argument, and
   * use the ParameterHandler::add_parameter() method to add parameters.
   */
  template <class SourceClass>
  class Proxy : public dealii::ParameterAcceptor, public SourceClass
  {
  public:
    /**
     * Default constructor. The argument `section_name` is forwarded to the
     * constructor of the ParameterAcceptor class, while all other arguments
     * are passed to the SourceClass constructor.
     */
    template <typename... Args>
    Proxy(const std::string &section_name, Args... args);
  };

  template <class SourceClass>
  template <typename... Args>
  Proxy<SourceClass>::Proxy(const std::string &section_name, Args... args)
    : dealii::ParameterAcceptor(section_name)
    , SourceClass(args...)
  {
    enter_my_subsection(this->prm);
    SourceClass::add_parameters(this->prm);
    leave_my_subsection(this->prm);
  }
} // namespace ParsedTools
#endif