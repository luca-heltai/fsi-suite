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


#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/cgal/intersections.h>

#include <set>
#include <tuple>
#include <vector>

#include "moonolith_tools.h"

using namespace dealii;

#if defined DEAL_II_PREFER_CGAL_OVER_PARMOONOLITH

namespace dealii
{
  namespace NonMatching
  {
    /**
     * Compute intersection between two deal.II cells, up to a user-defined
     * treshold that defaults to 1e-9.
     */
    template <int dim0, int dim1, int spacedim>
    dealii::Quadrature<spacedim>
    compute_cell_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
      const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
      const unsigned int                                           degree,
      const Mapping<dim0, spacedim>                               &mapping0,
      const Mapping<dim1, spacedim>                               &mapping1,
      const double                                                 tol = 1e-9)
    {
      if constexpr (dim0 == 1 && dim1 == 1)
        {
          (void)cell0;
          (void)cell1;
          (void)degree;
          (void)mapping0;
          (void)mapping1;
          (void)tol;
          AssertThrow(false, ExcNotImplemented());
          return dealii::Quadrature<spacedim>();
        }
      else
        {
          const auto &vec_arrays =
            ::CGALWrappers::compute_intersection_of_cells(
              cell0, cell1, mapping0, mapping1, tol);
          return QGaussSimplex<dim1>(degree).mapped_quadrature(vec_arrays);
        }
    }



#elif defined DEAL_II_WITH_PARMOONOLITH

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim>
    dealii::Quadrature<spacedim>
    compute_cell_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
      const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
      const unsigned int                                           degree,
      const Mapping<dim0, spacedim>                               &mapping0,
      const Mapping<dim1, spacedim>                               &mapping1)
    {
      if constexpr ((dim0 == 1 && dim1 == 3) || (dim0 == 3 && dim1 == 1) ||
                    (dim0 == 1 && dim1 == 1))
        {
          (void)cell0;
          (void)cell1;
          (void)degree;
          (void)mapping0;
          (void)mapping1;
          AssertThrow(false, ExcNotImplemented());
          return dealii::Quadrature<spacedim>();
        }
      else
        {
          return moonolith::compute_intersection(
            cell0, cell1, degree, mapping0, mapping1);
        }
    }
#else

namespace dealii
{
  namespace NonMatching
  {
    template <int dim0, int dim1, int spacedim>
    Quadrature<spacedim>
    compute_intersection(
      const typename Triangulation<dim0, spacedim>::cell_iterator &,
      const typename Triangulation<dim1, spacedim>::cell_iterator &,
      const unsigned int,
      const Mapping<dim0, spacedim> &,
      const Mapping<dim1, spacedim> &)
    {
      Assert(false,
             ExcMessage(
               "This function needs CGAL or PARMOONOLITH to be installed, "
               "but cmake could not find any of them."));
      return Quadrature<spacedim>();
    }

#endif

    template <int dim0, int dim1, int spacedim>
    std::vector<
      std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                 typename Triangulation<dim1, spacedim>::cell_iterator,
                 Quadrature<spacedim>>>
    compute_intersection(const GridTools::Cache<dim0, spacedim> &space_cache,
                         const GridTools::Cache<dim1, spacedim> &immersed_cache,
                         const unsigned int                      degree,
                         const double                            tol)
    {
      Assert(degree >= 1, ExcMessage("degree cannot be less than 1"));

      std::vector<
        std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                   typename Triangulation<dim1, spacedim>::cell_iterator,
                   Quadrature<spacedim>>>
        cells_with_quads;


      const auto &space_tree =
        space_cache.get_locally_owned_cell_bounding_boxes_rtree();

      // The immersed tree *must* contain all cells, also the non-locally owned
      // ones.
      const auto &immersed_tree =
        immersed_cache.get_cell_bounding_boxes_rtree();

      // references to triangulations' info (cp cstrs marked as delete)
      const auto &mapping0 = space_cache.get_mapping();
      const auto &mapping1 = immersed_cache.get_mapping();
      namespace bgi        = boost::geometry::index;
      // Whenever the BB space_cell intersects the BB of an embedded cell,
      // store the space_cell in the set of intersected_cells
      for (const auto &[immersed_box, immersed_cell] : immersed_tree)
        {
          for (const auto &[space_box, space_cell] :
               space_tree |
                 bgi::adaptors::queried(bgi::intersects(immersed_box)))
            {
              const auto &test_intersection =
                compute_cell_intersection<dim0, dim1, spacedim>(
                  space_cell, immersed_cell, degree, mapping0, mapping1);

              // if (test_intersection.get_points().size() !=
              const auto  &weights = test_intersection.get_weights();
              const double area =
                std::accumulate(weights.begin(), weights.end(), 0.0);
              if (area > tol) // non-trivial intersection
                {
                  cells_with_quads.push_back(std::make_tuple(
                    space_cell, immersed_cell, test_intersection));
                }
            }
        }

      return cells_with_quads;
    }


    template Quadrature<1>
    compute_cell_intersection(const Triangulation<1, 1>::cell_iterator &,
                              const Triangulation<1, 1>::cell_iterator &,
                              const unsigned int,
                              const Mapping<1, 1> &,
                              const Mapping<1, 1> &,
                              const double);


    template Quadrature<2>
    compute_cell_intersection(const Triangulation<2, 2>::cell_iterator &,
                              const Triangulation<1, 2>::cell_iterator &,
                              const unsigned int,
                              const Mapping<2, 2> &,
                              const Mapping<1, 2> &,
                              const double);

    template Quadrature<2>
    compute_cell_intersection(const Triangulation<2, 2>::cell_iterator &,
                              const Triangulation<2, 2>::cell_iterator &,
                              const unsigned int,
                              const Mapping<2, 2> &,
                              const Mapping<2, 2> &,
                              const double);


    template Quadrature<3>
    compute_cell_intersection(const Triangulation<3, 3>::cell_iterator &,
                              const Triangulation<2, 3>::cell_iterator &,
                              const unsigned int,
                              const Mapping<3, 3> &,
                              const Mapping<2, 3> &,
                              const double);

    template Quadrature<3>
    compute_cell_intersection(const Triangulation<3, 3>::cell_iterator &,
                              const Triangulation<3, 3>::cell_iterator &,
                              const unsigned int,
                              const Mapping<3, 3> &,
                              const Mapping<3, 3> &,
                              const double);

    template std::vector<
      std::tuple<typename dealii::Triangulation<1, 1>::cell_iterator,
                 typename dealii::Triangulation<1, 1>::cell_iterator,
                 Quadrature<1>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<1, 1> &space_cache,
      const GridTools::Cache<1, 1> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<
      std::tuple<typename dealii::Triangulation<3, 3>::cell_iterator,
                 typename dealii::Triangulation<1, 3>::cell_iterator,
                 Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<1, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);


    template std::vector<
      std::tuple<typename dealii::Triangulation<2, 2>::cell_iterator,
                 typename dealii::Triangulation<1, 2>::cell_iterator,
                 Quadrature<2>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<2, 2> &space_cache,
      const GridTools::Cache<1, 2> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<2, 2>::cell_iterator,
                                    typename Triangulation<2, 2>::cell_iterator,
                                    Quadrature<2>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<2, 2> &space_cache,
      const GridTools::Cache<2, 2> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                    typename Triangulation<2, 3>::cell_iterator,
                                    Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<2, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);



    template std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                    typename Triangulation<3, 3>::cell_iterator,
                                    Quadrature<3>>>
    NonMatching::compute_intersection(
      const GridTools::Cache<3, 3> &space_cache,
      const GridTools::Cache<3, 3> &immersed_cache,
      const unsigned int            degree,
      const double                  tol);
  }
}
