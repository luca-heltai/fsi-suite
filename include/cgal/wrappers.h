#ifndef dealii_cgal_wrappers_h
#define dealii_cgal_wrappers_h

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_CGAL

#  include <boost/config.hpp>

#  include <CGAL/Bbox_2.h>
#  include <CGAL/Bbox_3.h>
#  include <CGAL/Cartesian.h>
#  include <CGAL/Dimension.h>
#  include <CGAL/IO/io.h>
#  include <CGAL/Origin.h>
#  include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#  include <CGAL/Polyhedron_3.h>
#  include <CGAL/Polyhedron_incremental_builder_3.h>
#  include <CGAL/Simple_cartesian.h>
#  include <CGAL/Surface_mesh.h>


namespace CGALWrappers
{
  /**
   * Convert from deal.II Point to any compatible CGAL point.
   *
   * @tparam CGALPointType Any of the CGAL point types
   * @tparam dim Dimension of the point
   * @param [in] p An input deal.II Point<dim>
   * @return CGALPointType A CGAL point
   */
  template <typename CGALPointType, int dim>
  inline CGALPointType
  to_cgal(const dealii::Point<dim> &p);

  /**
   * Convert from various CGAL point types to deal.II Point.
   * @tparam dim Dimension of the point
   * @tparam CGALPointType Any of the CGAL point types
   * @param p
   * @return dealii::Point<dim>
   */
  template <int dim, typename CGALPointType>
  inline dealii::Point<dim>
  to_dealii(const CGALPointType &p);

  /**
   * Build a CGAL Polyhedron from a deal.II cell.
   *
   * If the @p poly argument is not null, the cell is appended to the existing
   * polyhedrons in @p poly.
   *
   * @tparam PolyhedronType A compatible with CGAL polyhedron
   * @tparam dim Dimension of the cell
   * @tparam spacedim Dimension of the embedding space
   * @param poly The output polyhedron
   * @param cell The input deal.II cell
   */
  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    PolyhedronType &                                                    poly);

  /**
   * Given a deal.II Triangulation, return the corresponding CGAL Polyhedron.
   *
   * @tparam PolyhedronType
   * @tparam dim
   * @tparam spacedim
   * @param tria
   * @param mapping
   * @param poly
   */
  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal(const typename dealii::Triangulation<dim, spacedim> &tria,
          const dealii::Mapping<dim, spacedim> &               mapping,
          PolyhedronType &                                     poly);

  namespace internal
  {
    /**
     * A CGAL modifier, used to create a polyhedron from a deal.II cell.
     * @tparam HDS A halfedge data structure compatible with CGAL
     * @tparam dim Dimension of the cell
     * @tparam spacedim Dimension of the embedding space
     */
    template <class HDS, int dim, int spacedim>
    class BuildCell : public CGAL::Modifier_base<HDS>
    {
    public:
      BuildCell(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator
          &                                   cell,
        const dealii::Mapping<dim, spacedim> &mapping)
        : cell(cell)
        , mapping(mapping)
      {}
      void
      operator()(HDS &hds)
      {
        // constexpr unsigned int dim = CellIterator::AccessorType::dimension;
        // Postcondition: hds is a valid polyhedral surface.
        CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
        typedef typename HDS::Vertex                Vertex;
        typedef typename Vertex::Point              CGALPoint;

        const auto vertices = mapping.get_vertices(cell);

        auto add_vertices = [&]() {
          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
            B.add_vertex(CGALWrappers::to_cgal<CGALPoint>(vertices[i]));
        };

        auto add_facet = [&](const std::vector<unsigned int> &facet) {
          B.add_facet(facet.begin(), facet.end());
        };

        switch (cell->n_vertices())
          {
            case 2:
              B.begin_surface(cell->n_vertices(), 1);
              add_vertices();
              add_facet({0, 1, 0});
              B.end_surface();
              break;
            case 3:
              B.begin_surface(cell->n_vertices(), 1);
              add_vertices();
              add_facet({0, 1, 2});
              B.end_surface();
              break;
            case 4:
              if constexpr (dim == 2)
                {
                  B.begin_surface(cell->n_vertices(), 1);
                  add_vertices();
                  add_facet({0, 1, 3, 2});
                  B.end_surface();
                }
              else
                {
                  B.begin_surface(cell->n_vertices(), 4);
                  add_vertices();
                  add_facet({0, 1, 2});
                  add_facet({1, 0, 3});
                  add_facet({2, 1, 3});
                  add_facet({0, 2, 3});
                  B.end_surface();
                }
              break;
            case 5:
              B.begin_surface(cell->n_vertices(), 5);
              add_vertices();
              add_facet({0, 1, 3, 2});
              add_facet({1, 0, 4});
              add_facet({3, 1, 4});
              add_facet({2, 3, 4});
              add_facet({0, 2, 4});
              B.end_surface();
              break;
            case 6:
              B.begin_surface(cell->n_vertices(), 5);
              add_vertices();
              add_facet({0, 1, 2});
              add_facet({1, 0, 3, 4});
              add_facet({1, 4, 5, 2});
              add_facet({3, 0, 2, 5});
              add_facet({4, 3, 5});
              B.end_surface();
              break;
            case 8:
              B.begin_surface(cell->n_vertices(), 6);
              add_vertices();
              add_facet({0, 1, 3, 2});
              add_facet({1, 0, 4, 5});
              add_facet({3, 1, 5, 7});
              add_facet({2, 3, 7, 6});
              add_facet({4, 0, 2, 6});
              add_facet({5, 4, 6, 7});
              B.end_surface();
              break;
            default:
              dealii::ExcInternalError();
          }
      }

    private:
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell;
      const dealii::Mapping<dim, spacedim> &mapping;
    };
  } // namespace internal

#  ifndef DOXYGEN
  // Template implementations
  template <typename CGALPointType, int dim>
  inline CGALPointType
  to_cgal(const dealii::Point<dim> &p)
  {
    constexpr int cdim = CGALPointType::Ambient_dimension::value;
    static_assert(dim <= cdim, "Only dim <= cdim supported");
    if constexpr (cdim == 1)
      return CGALPointType(p[0]);
    else if constexpr (cdim == 2)
      return CGALPointType(p[0], dim > 1 ? p[1] : 0);
    else if constexpr (cdim == 3)
      return CGALPointType(p[0], dim > 1 ? p[1] : 0, dim > 2 ? p[2] : 0);
    else
      Assert(false, dealii::ExcNotImplemented());
    return CGALPointType();
  }



  template <int dim, typename CGALPointType>
  inline dealii::Point<dim>
  to_dealii(const CGALPointType &p)
  {
    if constexpr (dim == 1)
      return dealii::Point<dim>(CGAL::to_double(p.x()));
    else if constexpr (dim == 2)
      return dealii::Point<dim>(CGAL::to_double(p.x()), CGAL::to_double(p.y()));
    else if constexpr (dim == 3)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                CGAL::to_double(p.y()),
                                CGAL::to_double(p.z()));
    else
      Assert(false, dealii::ExcNotImplemented());
  }



  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    PolyhedronType &                                                    poly)
  {
    internal::BuildCell<typename PolyhedronType::HalfedgeDS, dim, spacedim>
      cell_builder(cell, mapping);
    poly.delegate(cell_builder);
  }



  /**
   * @brief Create a CGAL::Surface_mesh starting from a deal.II cell
   *
   * @tparam CGALPointType
   * @tparam dim
   * @tparam spacedim
   * @param cell
   * @param mapping
   * @param surface_mesh Surface_mesh to be filled
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &surface_mesh)
  {
    typedef CGAL::Surface_mesh<CGALPointType> Mesh;
    typedef typename Mesh::Vertex_index       Vertex;
    const unsigned int                        n_vertices = cell->n_vertices();
    const auto &        vertices = mapping.get_vertices(cell);
    std::vector<Vertex> v_descriptors(n_vertices);

    auto add_vertices = [&]() {
      for (unsigned int i = 0; i < n_vertices; ++i)
        {
          v_descriptors[i] = surface_mesh.add_vertex(
            CGALWrappers::to_cgal<CGALPointType>(vertices[i]));
        }
    };

    auto reorder_vertices =
      [&v_descriptors](const std::vector<unsigned int> &facet) {
        std::vector<Vertex> ordered_vertices(facet.size());
        for (unsigned int i = 0; i < facet.size(); ++i)
          {
            ordered_vertices[i] = v_descriptors[facet[i]];
          }
        return ordered_vertices;
      };

    auto add_facet = [&](const std::vector<unsigned int> &facet) {
      const auto &v = reorder_vertices(facet);
      auto        f = surface_mesh.add_face(v);
    };



    switch (n_vertices)
      {
        case 8:
          add_vertices();
          add_facet({0, 1, 3, 2});
          add_facet({1, 0, 4, 5});
          add_facet({3, 1, 5, 7});
          add_facet({2, 3, 7, 6});
          add_facet({4, 0, 2, 6});
          add_facet({5, 4, 6, 7});
          break;

        default:
          dealii::ExcInternalError();
          break;
      }

  }


#  endif

} // namespace CGALWrappers
#endif
#endif