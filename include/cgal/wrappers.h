#ifndef dealii_cgal_wrappers_h
#define dealii_cgal_wrappers_h

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
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
#  include <CGAL/Polygon_mesh_processing/stitch_borders.h>
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
   * Build a CGAL Surface_mesh from a deal.II cell.
   *
   * @tparam CGALPointType A point compatible with CGAL
   * @tparam dim Dimension of the cell
   * @tparam spacedim Dimension of the embedding space
   * @param mesh The output surface mesh
   * @param cell The input deal.II cell
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &                                 mesh);

  /**
   * @brief Create a CGAL::Surface_mesh starting from a deal.II triangulation.
   *
   * @param[in] tria Input triangulation
   * @param surface_mesh Surface_mesh to be filled
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(const dealii::Triangulation<dim, spacedim> &tria,
               const dealii::Mapping<dim, spacedim> &      mapping,
               CGAL::Surface_mesh<CGALPointType> &         surface_mesh);

  /**
   * Build a CGAL Polyhedron from a deal.II cell.
   *
   * If the @p poly argument is not null, the cell is appended to the existing
   * polyhedrons in @p poly.
   *
   * @param poly The output polyhedron
   * @param cell The input deal.II cell
   */
  template <class PolyhedronType, int dim, int spacedim>
  void
  to_cgal_poly(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    PolyhedronType &                                                    poly);

  /**
   * Given a deal.II Triangulation, return a CGAL Polyhedron mesh.
   *
   * Technically speacking, CGAL supports only surface meshes. This function
   * will generate a collection of disjoing polyhedron surface meshes from each
   * cell of the triangulation. Beware of the fact that the resulting mesh may
   * not be a valid CGAL polyhedral mesh, especially if you have a triangulation
   * that contains non-planar surfaces.
   *
   * If you want to work with the resulting polyhedral mesh you may want to call
   * the free function CGAL::Polygon_mesh_processing::triangulate_faces() on the
   * resulting @p poly object.
   *
   * @param[in] tria  Input dealii Triangulation
   * @param[in] mapping The mapping used to map the vertices of the
   * triangulation
   * @param[out] poly Output CGAL polyhedron
   */
  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal_poly(const typename dealii::Triangulation<dim, spacedim> &tria,
               const dealii::Mapping<dim, spacedim> &               mapping,
               PolyhedronType &                                     poly);

  /**
   * @brief Create a Triangulation starting from a cgal surface_mesh.
   *
   * @param[in] surface_mesh Input surface mesh
   * @param[out] tria Surface_mesh to be filled
   */
  template <typename CGALPointType, int spacedim>
  void
  to_dealii(const CGAL::Surface_mesh<CGALPointType> &surface_mesh,
            dealii::Triangulation<2, spacedim> &     tria);

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
      /**
       * Construct a polyhedron modifier from a dealii cell iterator.
       *
       * This class is used to add to a halfedge data structure the vertices and
       * the faces of a dealii cell.
       *
       * @param[in] cell The input deal.II cell iterator
       * @param[in] mapping The mapping used to map the vertices of the
       * triangulation
       */
      BuildCell(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator
          &                                   cell,
        const dealii::Mapping<dim, spacedim> &mapping)
        : cell(cell)
        , mapping(mapping)
      {}

      /**
       * Apply the modifier to the halfedge data structure.
       *
       * @param hds The halfedge data structure to modify.
       */
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
      /**
       * A const reference to the input cell.
       */
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell;

      /**
       * A const reference to the input mapping.
       */
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
    constexpr int cdim = CGALPointType::Ambient_dimension::value;
    if constexpr (dim == 1)
      return dealii::Point<dim>(CGAL::to_double(p.x()));
    else if constexpr (dim == 2)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                cdim > 1 ? CGAL::to_double(p.y()) : 0);
    else if constexpr (dim == 3)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                cdim > 1 ? CGAL::to_double(p.y()) : 0,
                                cdim > 2 ? CGAL::to_double(p.z()) : 0);
    else
      Assert(false, dealii::ExcNotImplemented());
  }



  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal_poly(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    PolyhedronType &                                                    poly)
  {
    internal::BuildCell<typename PolyhedronType::HalfedgeDS, dim, spacedim>
      cell_builder(cell, mapping);
    poly.delegate(cell_builder);
  }



  template <typename PolyhedronType, int dim, int spacedim>
  void
  to_cgal_poly(const dealii::Triangulation<dim, spacedim> &tria,
               const dealii::Mapping<dim, spacedim> &      mapping,
               PolyhedronType &                            poly)
  {
    for (const auto &cell : tria.active_cell_iterators())
      to_cgal_poly(cell, mapping, poly);
    CGAL::Polygon_mesh_processing::stitch_borders(poly);
  }



  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &surface_mesh)
  {
    AssertThrow(CGALPointType::Ambient_dimension::value >= spacedim,
                dealii::ExcMessage("The CGAL mesh must be able to store " +
                                   std::to_string(spacedim) +
                                   "-dimensional points"));

    CGAL::Polyhedron_3<CGAL::Simple_cartesian<double>> poly;
    to_cgal_poly(cell, mapping, poly);
    CGAL::copy_face_graph(poly, surface_mesh);
  }



  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(const typename dealii::Triangulation<dim, spacedim> &tria,
               const dealii::Mapping<dim, spacedim> &               mapping,
               CGAL::Surface_mesh<CGALPointType> &surface_mesh)
  {
    AssertThrow(CGALPointType::Ambient_dimension::value >= spacedim,
                dealii::ExcMessage("The CGAL mesh must be able to store " +
                                   std::to_string(spacedim) +
                                   "-dimensional points"));

    CGAL::Polyhedron_3<CGAL::Simple_cartesian<double>> poly;
    to_cgal_poly(tria, mapping, poly);
    CGAL::copy_face_graph(poly, surface_mesh);
  }



  template <typename CGALPointType, int spacedim>
  void
  to_dealii(const CGAL::Surface_mesh<CGALPointType> &surface_mesh,
            dealii::Triangulation<2, spacedim> &     tria)
  {
    AssertThrow(CGALPointType::Ambient_dimension::value <= spacedim,
                dealii::ExcMessage(
                  "The dealii mesh must be able to store " +
                  std::to_string(CGALPointType::Ambient_dimension::value) +
                  "-dimensional points"));

    std::vector<dealii::Point<spacedim>> vertices;
    std::vector<dealii::CellData<2>>     cells;
    dealii::SubCellData                  subcells;

    vertices.reserve(surface_mesh.num_vertices());
    for (const auto &v : surface_mesh.points())
      vertices.emplace_back(CGALWrappers::to_dealii<spacedim>(v));

    for (const auto &face : surface_mesh.faces())
      {
        std::vector<unsigned int> cell_vertices;
        for (const auto v :
             CGAL::vertices_around_face(surface_mesh.halfedge(face),
                                        surface_mesh))
          cell_vertices.push_back(v);

        if (cell_vertices.size() == 4)
          std::swap(cell_vertices[3], cell_vertices[2]);

        dealii::CellData<2> c(cell_vertices.size());
        std::swap(c.vertices, cell_vertices);
        cells.emplace_back(c);
      }
    tria.create_triangulation(vertices, cells, subcells);
  }
#  endif

} // namespace CGALWrappers
#endif
#endif