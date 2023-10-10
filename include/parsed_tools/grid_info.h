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

#ifndef parsed_tools_grid_info_h
#define parsed_tools_grid_info_h

#include <deal.II/base/patterns.h>

#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

namespace dealii
{
  namespace Patterns
  {
    namespace Tools
    {
      /**
       * @brief Instruct deal.II on how to convert a manifold_id to a string.
       *
       * @tparam
       */
      template <>
      struct Convert<dealii::types::manifold_id>
      {
        using T = dealii::types::manifold_id;
        /**
         * @brief Default pattern for converting a manifold_id to a string.
         *
         * @return std::unique_ptr<dealii::Patterns::PatternBase>
         */
        static std::unique_ptr<dealii::Patterns::PatternBase>
        to_pattern()
        {
          return Patterns::Integer(-1).clone();
        }

        /**
         * @brief Convert a manifold_id to a string.
         *
         * @param t The manifold_id to convert.
         * @param pattern Optional pattern to use.
         * @return std::string The string representation of the manifold_id.
         */
        static std::string
        to_string(const T &                            t,
                  const dealii::Patterns::PatternBase &pattern =
                    *Convert<T>::to_pattern())
        {
          return Convert<int>::to_string(static_cast<int>(t), pattern);
        }

        /**
         * @brief Convert a string to a manifold_id.
         *
         * @param s The string to convert.
         * @param pattern Optional pattern to use.
         * @return T The manifold_id.
         */
        static T
        to_value(const std::string &                  s,
                 const dealii::Patterns::PatternBase &pattern =
                   *Convert<T>::to_pattern())
        {
          return T(Convert<int>::to_value(s, pattern));
        }
      };

      /**
       * @brief Instruct deal.II on how to convert a dealii::ReferenceCell to a string.
       *
       * @tparam
       */
      template <>
      struct Convert<dealii::ReferenceCell>
      {
        using T = dealii::ReferenceCell;
        /**
         * @brief  Default pattern for converting a dealii::ReferenceCell to a string.
         *
         * @return std::unique_ptr<dealii::Patterns::PatternBase>
         */
        static std::unique_ptr<dealii::Patterns::PatternBase>
        to_pattern()
        {
          return Convert<int>::to_pattern();
        }

        /**
         * @brief Convert a dealii::ReferenceCell to a string.
         *
         * @param t The dealii::ReferenceCell to convert.
         * @param pattern Optional pattern to use.
         * @return std::string The string representation of the dealii::ReferenceCell.
         */
        static std::string
        to_string(const T &                            t,
                  const dealii::Patterns::PatternBase &pattern =
                    *Convert<T>::to_pattern())
        {
          return Convert<int>::to_string(t, pattern);
        }


        /**
         * @brief Convert a string to a dealii::ReferenceCell.
         *
         * @param s The string to convert.
         * @param pattern Optional pattern to use.
         * @return T The dealii::ReferenceCell.
         */
        static T
        to_value(const std::string &                  s,
                 const dealii::Patterns::PatternBase &pattern =
                   *Convert<T>::to_pattern())
        {
          return dealii::internal::ReferenceCell::make_reference_cell_from_int(
            Convert<int>::to_value(s, pattern));
        }
      };
    } // namespace Tools
  }   // namespace Patterns
} // namespace dealii

namespace ParsedTools
{
  /**
   * @brief Gather information about a Triangulation.
   *
   * This class can be used to store information about a Triangulation, such as
   * the number of cells, the number of vertices, etc.
   *
   * It is particularly useful when building robust programs w.r.t. boundary
   * conditions, boundary indicators, manifold indicators, etc.
   */
  struct GridInfo
  {
    /**
     * @brief Construct a new (empty) Grid Info object
     *
     * @param info_level Level of information to gather. The higher the number,
     * the more expensive the operation.
     */
    GridInfo(const unsigned int info_level = 0)
      : info_level(info_level)
    {}

    /**
     * @brief Construct a new Grid Info object, and gather all information about
     * the Triangulation @p tria
     *
     * @param tria The Triangulation to gather information from.
     * @param info_level Level of information to gather. The higher the number,
     * the more expensive the operation.
     */
    template <int dim, int spacedim>
    GridInfo(const dealii::Triangulation<dim, spacedim> &tria,
             const unsigned int                          info_level = 0)
      : info_level(info_level)
    {
      build_info(tria);
    }

    /**
     * @brief Actually build the information about the Triangulation.
     *
     * @param tria The Triangulation to gather information from.
     */
    template <int dim, int spacedim>
    void
    build_info(const dealii::Triangulation<dim, spacedim> &tria)
    {
      n_active_cells  = tria.n_active_cells();
      n_vertices      = tria.n_vertices();
      n_used_vertices = tria.n_used_vertices();
      n_levels        = tria.n_levels();
      if (info_level > 0)
        {
          n_active_cells_at_level.resize(n_levels);
          n_cells_at_level.resize(n_levels);
          for (unsigned int i = 0; i < n_levels; ++i)
            {
              n_active_cells_at_level[i] = tria.n_active_cells(i);
              n_cells_at_level[i]        = tria.n_cells(i);
            }
        }
      if (info_level > 1)
        {
          boundary_ids         = tria.get_boundary_ids();
          manifold_ids         = tria.get_manifold_ids();
          reference_cell_types = tria.get_reference_cells();

          std::set<dealii::types::material_id> m_ids;
          for (const auto &cell : tria.active_cell_iterators())
            m_ids.insert(cell->material_id());
          material_ids.insert(material_ids.end(), m_ids.begin(), m_ids.end());
        }

      if (info_level > 2)
        {
          for (const auto &id : boundary_ids)
            faces_per_boundary_id[id] = 0;

          for (const auto &id : material_ids)
            cells_per_material_id[id] = 0;

          for (const auto &id : manifold_ids)
            {
              faces_per_manifold_id[id] = 0;
              cells_per_manifold_id[id] = 0;
            }

          for (const auto &id : reference_cell_types)
            cells_per_reference_cell_type[id] = 0;

          for (const auto &cell : tria.active_cell_iterators())
            {
              ++cells_per_material_id[cell->material_id()];
              ++cells_per_manifold_id[cell->manifold_id()];
              ++cells_per_reference_cell_type[cell->reference_cell()];
            }

          for (const auto &f : tria.active_face_iterators())
            {
              if (f->at_boundary())
                ++faces_per_boundary_id[f->boundary_id()];
              ++faces_per_manifold_id[f->manifold_id()];
            }
        }
    }

    /**
     * @brief Print all gathered information about the Triangulation.
     *
     * @param out The stream to print to.
     */
    template <typename StreamType>
    void
    print_info(StreamType &out)
    {
      out << "Active cells  : " << n_active_cells << std::endl
          << "Vertices      : " << n_vertices << std::endl
          << "Used vertices : " << n_used_vertices << std::endl
          << "Levels        : " << n_levels << std::endl;
      if (info_level > 0 && n_levels > 1)
        {
          out << "Active cells/level  : "
              << dealii::Patterns::Tools::to_string(n_active_cells_at_level)
              << std::endl
              << "Cells/level         : "
              << dealii::Patterns::Tools::to_string(n_cells_at_level)
              << std::endl;
        }
      if (info_level > 1)
        {
          out << "Boundary ids         : "
              << dealii::Patterns::Tools::to_string(boundary_ids) << std::endl
              << "Manifold ids         : "
              << dealii::Patterns::Tools::to_string(manifold_ids) << std::endl
              << "Material ids         : "
              << dealii::Patterns::Tools::to_string(material_ids) << std::endl
              << "Reference cell types : "
              << dealii::Patterns::Tools::to_string(reference_cell_types)
              << std::endl;
        }
      if (info_level > 2)
        {
          out << "Boundary id:n_faces         : "
              << dealii::Patterns::Tools::to_string(faces_per_boundary_id)
              << std::endl
              << "Material id:n_cells         : "
              << dealii::Patterns::Tools::to_string(cells_per_material_id)
              << std::endl
              << "Manifold id:n_faces         : "
              << dealii::Patterns::Tools::to_string(faces_per_manifold_id)
              << std::endl
              << "Manifold id:n_cells         : "
              << dealii::Patterns::Tools::to_string(cells_per_manifold_id)
              << std::endl
              << "Reference cell type:n_cells : "
              << dealii::Patterns::Tools::to_string(
                   cells_per_reference_cell_type)
              << std::endl;
        }
    }

    /** Level of information to gather. */
    unsigned int info_level = 0;

    /** Number of active cells. */
    unsigned int n_active_cells = 0;

    /** Number of vertices. */
    unsigned int n_vertices = 0;

    /** Number of used vertices. */
    unsigned int n_used_vertices = 0;

    /** Number of levels. */
    unsigned int n_levels = 0;

    /** Number of active cells at each level. */
    std::vector<unsigned int> n_active_cells_at_level;

    /** Number of cells at each level. */
    std::vector<unsigned int> n_cells_at_level;

    /** Boundary ids. */
    std::vector<dealii::types::boundary_id> boundary_ids;

    /** Material ids. */
    std::vector<dealii::types::material_id> material_ids;

    /** Manifold ids. */
    std::vector<dealii::types::manifold_id> manifold_ids;

    /** Reference cell types. */
    std::vector<dealii::ReferenceCell> reference_cell_types;

    /** Number of faces per boundary id. */
    std::map<dealii::types::boundary_id, unsigned int> faces_per_boundary_id;

    /** Number of cells per boundary id. */
    std::map<dealii::types::material_id, unsigned int> cells_per_material_id;

    /** Number of faces per manifold id. */
    std::map<dealii::types::manifold_id, unsigned int> faces_per_manifold_id;

    /** Number of cells per manifold id. */
    std::map<dealii::types::manifold_id, unsigned int> cells_per_manifold_id;

    /** Number of cells per reference cell type. */
    std::map<dealii::ReferenceCell, unsigned int> cells_per_reference_cell_type;
  };
} // namespace ParsedTools
#endif
