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

#ifndef parsed_tools_components_h
#define parsed_tools_components_h

#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include "parsed_tools/enum.h"

namespace ParsedTools
{
  namespace Components
  {
    /**
     * Enumerator used to identify patterns of components and their size in the
     * a block system.
     */
    enum class Type
    {
      all        = 0, //!< All components
      scalar     = 1, //!< Scalar
      vector     = 2, //!< Vector
      normal     = 3, //!< Normal component of a vector
      tangential = 4, //!< Tangential component of a vector
    };

    /**
     * Get a Lagrangian FiniteElement object compatible with the given
     * Triangulation object.
     *
     * The returned object can be either FE_Q, FE_DGQ, FE_P, or FE_DGP according
     * to the types of elements of the Triangulation (quads/hexes VS tria/tets)
     *
     * @param tria The triangulation to inspect
     * @param degree The polynomial degree of the Lagrangian FiniteElement
     * @param continuity The continuity of the Lagrangian FiniteElement: 0 for
     * continuous, -1 for discontinuous
     *
     * @return std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
     */
    template <int dim, int spacedim>
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
    get_lagrangian_finite_element(
      const dealii::Triangulation<dim, spacedim> &tria,
      const unsigned int                          degree     = 1,
      const int                                   continuity = 0);

    /**
     * Return a Quadrature object that can be used on the given Triangulation
     * cells.
     *
     * @param tria Triangulation to insepct
     * @param degree The degree of the 1d quadrature used to generate the actual
     * quadrature.
     * @return dealii::Quadrature<dim>
     */
    template <int dim, int spacedim>
    dealii::Quadrature<dim>
    get_cell_quadrature(const dealii::Triangulation<dim, spacedim> &tria,
                        const unsigned int                          degree);

    /**
     * Return a Quadrature object that can be used on the given Triangulation
     * faces.
     *
     * @param tria Triangulation to insepct
     * @param degree The degree of the 1d quadrature used to generate the actual
     * quadrature.
     * @return dealii::Quadrature<dim>
     */
    template <int dim, int spacedim>
    dealii::Quadrature<dim - 1>
    get_face_quadrature(const dealii::Triangulation<dim, spacedim> &tria,
                        const unsigned int                          degree);

    /**
     * Join strings in a container together using
     * a given separator.
     */
    template <typename Container>
    std::string
    join(const Container &strings, const std::string &separator);

    /**
     * Count the number of components in the given list of comma separated
     * components.
     */
    unsigned int
    n_components(const std::string &component_names);

    /**
     * Count the number of components in the given list of comma separated
     * components.
     */
    unsigned int
    n_blocks(const std::string &component_names);


    /**
     * Build component names from block names and multiplicities.
     */
    std::string
    blocks_to_names(const std::vector<std::string> & components,
                    const std::vector<unsigned int> &multiplicities);

    /**
     * Build block names and multiplicities from component names.
     */
    std::pair<std::vector<std::string>, std::vector<unsigned int>>
    names_to_blocks(const std::string &component_names);

    /**
     * Return the indices within the block corresponding to the given selected
     * components, or numbers::invalid_unsigned_int if the selected components
     * could not be found.
     *
     * Accepted patterns are comma separated versions of the following string
     * types:
     * - single component names, e.g., "p"
     * - block names, e.g., "u,p"
     * - component indices, e.g., "0,1"
     * - normal component names, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = block_indices(names, "u"); // {0}
     * r = block_indices(names, "u, p"); // {0,1}
     * r = block_indices(names, "p"); // {1}
     * r = block_indices(names, "2"); // {1}
     * r = block_indices(names, "1"); // {0}
     * r = block_indices(names, "0,1,2"); // {0,0,1}
     * r = block_indices(names, "u.n"); // {0}
     * r = block_indices(names, "u.t"); // {0}
     * @endcode
     */
    std::vector<unsigned int>
    block_indices(const std::string &component_names,
                  const std::string &selected_components);

    /**
     * Return the indices within the components corresponding to first component
     * of the given selected components, or numbers::invalid_unsigned_int if the
     * selected components could not be found.
     *
     * Accepted patterns are comma separated versions of the following string
     * types:
     * - single component names, e.g., "p"
     * - block names, e.g., "u,p"
     * - component indices, e.g., "0,1"
     * - normal component names, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_indices(names, "u"); // {0}
     * r = component_indices(names, "u, p"); // {0,2}
     * r = component_indices(names, "p"); // {2}
     * r = component_indices(names, "2"); // {2}
     * r = component_indices(names, "1"); // {0}
     * r = component_indices(names, "0,1,2"); // {0,0,1}
     * r = component_indices(names, "u.n"); // {0}
     * r = component_indices(names, "u.T"); // {0}
     * @endcode
     *
     * If the component names are unique, this is the same as the
     * block_indices() function, otherwise it returns the first component
     * indices.
     */
    std::vector<unsigned int>
    component_indices(const std::string &component_names,
                      const std::string &selected_components);

    /**
     * Return the indices within the components and within the blocks
     * corresponding to first component of the given selected component, or
     * numbers::invalid_unsigned_int if the selected components could not be
     * found.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the indices of the first component of the selected
     * component, and the index of the block, so, for "u" it would return {0,0}
     * and for "p" it would return {2,1} (third component, second block).
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     * - tangent component names, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_to_block_indices_map(names, "u"); // {0,0}
     * r = component_to_block_indices_map(names, "p"); // {2,1}
     * r = component_to_block_indices_map(names, "p"); // {2}
     * r = component_to_block_indices_map(names, "2"); // {2}
     * r = component_to_block_indices_map(names, "1"); // {0}
     * r = component_to_block_indices_map(names, "0,1,2"); // {0,0,1}
     * r = component_to_block_indices_map(names, "u.n"); // {0}
     * r = component_to_block_indices_map(names, "u.t"); // {0}
     * @endcode
     */
    std::pair<unsigned int, unsigned int>
    component_to_indices(const std::string &component_names,
                         const std::string &selected_component);

    /**
     * Return the canonical component name for the given selected component.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the canonical component name for an associated
     * pattern of components.
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = component_name(names, "u"); // "u"
     * r = component_name(names, "p"); // "p"
     * r = component_name(names, "1"); // "u"
     * r = component_name(names, "2"); // "p"
     * r = component_name(names, "u.n"); // "u"
     * @endcode
     */
    std::string
    component_name(const std::string &component_names,
                   const std::string &selected_component);

    /**
     * Return the component type for the given selected component.
     *
     * For example, for component names "u,u,p", there are three components (two
     * for "u" and one for "p"), and two blocks ("u" and "p").
     *
     * This function returns the component type for an associated pattern of
     * components, according to the multiplicity of the given component, and to
     * the normal or tangential component.
     *
     * Accepted patterns are the following string types:
     * - single component or block names, e.g., "p"
     * - a single component index, e.g., "0"
     * - a normal component name, e.g., "u.n" or "u.N"
     * - a tangential component name, e.g., "u.t" or "u.T"
     *
     * Examples:
     * @code
     * std::string names="u,u,p";
     * auto r = type(names, "u"); // vector
     * r = type(names, "p"); // scalar
     * r = type(names, "1"); // vector
     * r = type(names, "2"); // scalar
     * r = type(names, "u.n"); // normal
     * r = type(names, "u.t"); // tangential
     * @endcode
     */
    Type
    type(const std::string &component_name,
         const std::string &selected_component);

    /**
     * Return a component mask corresponding to a given selected component, from
     * a list of comma separated components.
     */
    dealii::ComponentMask
    mask(const std::string &component_names,
         const std::string &selected_component);



#ifndef DOXYGEN
    // Template implementation
    template <typename Container>
    std::string
    join(const Container &strings, const std::string &separator)
    {
      std::string result;
      std::string sep = "";
      for (const auto &s : strings)
        {
          result += sep + s;
          sep = separator;
        }
      return result;
    }



    template <int dim, int spacedim>
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
    get_lagrangian_finite_element(
      const dealii::Triangulation<dim, spacedim> &tria,
      const unsigned int                          degree,
      const int                                   continuity)
    {
      const auto ref_cells = tria.get_reference_cells();
      AssertThrow(
        ref_cells.size() == 1,
        dealii::ExcMessage(
          "This function does nots support mixed simplx/hex grid types."));
      AssertThrow(continuity == -1 || continuity == 0,
                  dealii::ExcMessage(
                    "only -1 and 0 are supported for continuity"));
      std::unique_ptr<dealii::FiniteElement<dim, spacedim>> result;
      if (ref_cells[0].is_simplex())
        {
          if (continuity == 0)
            result.reset(new dealii::FE_SimplexP<dim, spacedim>(degree));
          else
            result.reset(new dealii::FE_SimplexDGP<dim, spacedim>(degree));
        }
      else
        {
          Assert(ref_cells[0].is_hyper_cube(),
                 dealii::ExcMessage(
                   "Only simplex and hex cells are supported"));
          if (continuity == 0)
            result.reset(new dealii::FE_Q<dim, spacedim>(degree));
          else
            result.reset(new dealii::FE_DGQ<dim, spacedim>(degree));
        }
      return result;
    }



    template <int dim, int spacedim>
    dealii::Quadrature<dim>
    get_cell_quadrature(const dealii::Triangulation<dim, spacedim> &tria,
                        const unsigned int                          degree)
    {
      const auto ref_cells = tria.get_reference_cells();

      AssertThrow(
        ref_cells.size() == 1,
        dealii::ExcMessage(
          "This function does nots support mixed simplx/hex grid types."));
      return ref_cells[0].template get_gauss_type_quadrature<dim>(degree);
    }



    template <int dim, int spacedim>
    dealii::Quadrature<dim - 1>
    get_face_quadrature(const dealii::Triangulation<dim, spacedim> &tria,
                        const unsigned int                          degree)
    {
      if constexpr (dim == 1)
        {
          return dealii::QGauss<dim - 1>(degree);
        }
      else
        {
          const auto ref_cells = tria.get_reference_cells();

          AssertThrow(
            ref_cells.size() == 1,
            dealii::ExcMessage(
              "This function does nots support mixed simplx/hex grid types."));

          const dealii::ReferenceCell face_type =
            ref_cells[0].face_reference_cell(0);
          return face_type.template get_gauss_type_quadrature<dim - 1>(degree);
        }
    }
#endif
  } // namespace Components
} // namespace ParsedTools
#endif