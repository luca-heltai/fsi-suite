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

#ifndef parsed_tools_mapping_fe_field_h
#define parsed_tools_mapping_fe_field_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/numerics/vector_tools.h>

namespace ParsedTools
{
  /**
   * A wrapper class for MappingFEField or MappingQEulerian.
   *
   * This class can be used to create a mapping object that transforms the
   * triangulation it is applied to, according to a FiniteElement field, defined
   * by a vector based DoFHandler object and the corresponding field.
   *
   * This class is used in the following way:
   * @code{.cpp}
   * ParsedTools::MappingEulerian<dim, spacedim> mapping("/",
   *                                                    dof_handler,
   *                                                    "2*x; .5*y");
   * Vector<double> position(dof_handler.n_dofs());
   * mapping.initialize(position);
   *
   * // using mapping, now you can initialize an FEValues object which will see
   * // a grid which is strecthed by 2 in the horizontal
   * // direction, and compressed by 0.5 in the vertical direction
   * FEValues<dim, spacedim> fe_values(mapping, ....);
   * @endcode
   *
   * This is the base class to be used in elasticity problems, where the mapping
   * represents the actual physical configuration of the problem, or in ALE
   * problems, where the computational domain is mapped back to a reference
   * domain. In this case, the reference domain is the one stored in the
   * DoFHandler and Triangulation objects, while the computational domain is the
   * domain deformed according to the displacement vector used to initialize
   * this class.
   *
   * Upon construction, the class is not usable. Befor you can use it, you
   * should call the initialize() method, or the reset() method.
   *
   * Notice that the initialize() method actually sets the displacement vector
   * according to what is specified in the paramter file, while the reset()
   * method assumes that the vector is already initialized to the correct
   * values, and simply resets the internal dealii::MappingEulerian or
   * dealii::MappingQEulerian objects.
   */
  template <int dim, int spacedim = dim>
  class MappingEulerian : public dealii::ParameterAcceptor
  {
  public:
    MappingEulerian(
      const dealii::DoFHandler<dim, spacedim> &dh,
      const std::string                       &section_name              = "",
      const std::string           &initial_configuration_or_displacement = "",
      const bool                   use_displacement = false,
      const dealii::ComponentMask &mask             = dealii::ComponentMask());

    /**
     * Actually build the mapping from the given configuration or displacement.
     * If the @p initial_configuration_or_displacement parameter is not empty,
     * the input vector is modified to interpolate the given configuration or
     * displacement, otherwise the identity configuration is used.
     */
    template <typename VectorType>
    void
    initialize(VectorType &configuration_or_displacement);


    /**
     * Actually build the mapping from the given configuration or displacement.
     * If the @p initial_configuration_or_displacement parameter is not empty,
     * the input vector is modified to interpolate the given configuration or
     * displacement, otherwise the identity configuration is used.
     */
    template <typename VectorType>
    void
    initialize(VectorType &configuration_or_displacement,
               VectorType &locally_relevant_configuration_or_displacement);

    /**
     * Act as mapping.
     */
    operator const dealii::Mapping<dim, spacedim> &() const;

    /**
     * Return a reference to the actual mapping.
     */
    const dealii::Mapping<dim, spacedim> &
    operator()() const;

  private:
    /**
     * A pointer to the dof handler.
     */
    const dealii::SmartPointer<const dealii::DoFHandler<dim, spacedim>>
      dof_handler;

    /**
     * What components should be interpreted as the displacement.
     */
    const dealii::ComponentMask mask;

    /**
     * Switch from displacement to configuration.
     */
    bool use_displacement;

    /**
     * What configuration to store when the mapping is initialized.
     */
    std::string initial_configuration_or_displacement_expression;

    /**
     * The actual mapping.
     */
    std::unique_ptr<dealii::Mapping<dim, spacedim>> mapping;
  };


// Template implementation
#ifndef DOXYGEN
  template <int dim, int spacedim>
  template <typename VectorType>
  void
  MappingEulerian<dim, spacedim>::initialize(
    VectorType &configuration_or_displacement)
  {
    initialize(configuration_or_displacement, configuration_or_displacement);
  }



  template <int dim, int spacedim>
  template <typename VectorType>
  void
  MappingEulerian<dim, spacedim>::initialize(
    VectorType &configuration_or_displacement,
    VectorType &locally_relevant_configuration_or_displacement)
  {
    // Check that the dofhandler is ok.
    AssertThrow(mask == dealii::ComponentMask() ||
                  spacedim == mask.n_selected_components(
                                dof_handler->get_fe().n_components()),
                dealii::ExcMessage("The number of selected components in the "
                                   "mask  must be " +
                                   std::to_string(spacedim) +
                                   ", it is instead " +
                                   std::to_string(mask.n_selected_components(
                                     dof_handler->get_fe().n_components()))));

    // Interpolate the user data
    if (initial_configuration_or_displacement_expression != "")
      {
        dealii::FunctionParser<spacedim> initial_configuration(
          initial_configuration_or_displacement_expression);

        AssertThrow(initial_configuration.n_components ==
                      dof_handler->get_fe().n_components(),
                    dealii::ExcMessage("The number of components in the "
                                       "configuration expression must be equal "
                                       "to the number of components in the "
                                       "fe."));


        dealii::VectorTools::interpolate(*dof_handler,
                                         initial_configuration,
                                         configuration_or_displacement,
                                         mask);
      }
    else
      {
        if (use_displacement)
          configuration_or_displacement = 0;
        else
          {
            if (dof_handler->get_triangulation()
                  .all_reference_cells_are_hyper_cube())
              dealii::VectorTools::get_position_vector(
                *dof_handler, configuration_or_displacement, mask);
            else
              {
                // Use interpolation to get the position vector.
                const std::string id[3] = {"x", "y", "z"};
                std::string       id_expression;
                std::string       sep = "";
                unsigned int      j   = 0;
                for (unsigned int i = 0;
                     i < dof_handler->get_fe().n_components();
                     ++i)
                  {
                    if (mask[i] && j < spacedim)
                      id_expression += sep + id[j++];
                    else
                      id_expression += sep + "0";
                    sep = "; ";
                  }

                dealii::FunctionParser<spacedim> initial_configuration(
                  id_expression);

                dealii::VectorTools::interpolate(*dof_handler,
                                                 initial_configuration,
                                                 configuration_or_displacement,
                                                 mask);
              }
          }
      }
    // Copy to the locally relevant vector
    locally_relevant_configuration_or_displacement =
      configuration_or_displacement;

    if (use_displacement)
      // Finallly initialize the mapping with our own configuration
      // vector.
      {
        AssertThrow(
          dof_handler->get_triangulation().all_reference_cells_are_hyper_cube(),
          dealii::ExcMessage("The displacement mapping is only "
                             "supported for hypercube grids."));

        AssertThrow(mask == dealii::ComponentMask(),
                    dealii::ExcMessage(
                      "Using the displacement is only possible with default "
                      "component mask."));

        const auto degree = dof_handler->get_fe().degree;
        mapping.reset(new dealii::MappingQEulerian<dim, VectorType, spacedim>(
          degree,
          *dof_handler,
          locally_relevant_configuration_or_displacement));
      }
    else
      {
        // Finallly initialize the mapping with our own configuration
        // vector.
        mapping.reset(new dealii::MappingFEField<dim, spacedim, VectorType>(
          *dof_handler, locally_relevant_configuration_or_displacement, mask));
      }
  }
#endif

} // namespace ParsedTools
#endif