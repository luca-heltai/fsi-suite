#include <deal.II/base/config.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

// #ifdef DEAL_II_WIHT_CGAL
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/IO/io.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_approximation/approximate_triangle_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/boost/graph/io.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/refine_mesh_3.h>

#include "cgal/wrappers.h"

// typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Simple_cartesian<double>                K;
typedef CGAL::Delaunay_triangulation_2<K>             Delaunay;
typedef Delaunay::Point                               DPoint;
typedef CGAL::Polyhedron_3<K>                         Polyhedron;
typedef Polyhedron::HalfedgeDS                        HalfedgeDS;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;


#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif
// Triangulation
typedef CGAL::
  Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr>                       C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;


TEST(CGAL, Delaunay2D)
{
  CGAL::Random_points_in_disc_2<DPoint> rnd;
  const unsigned int                    n_vertices = 60;

  std::vector<DPoint> points(n_vertices);

  for (unsigned int i = 0; i < n_vertices; ++i)
    points[i] = *rnd++;
  Delaunay cgal_tria(points.begin(), points.end());


  std::vector<dealii::Point<2>> vertices;
  std::vector<CellData<2>>      cells;
  SubCellData                   subcells;

  std::map<Delaunay::Vertex_handle, unsigned int> v_map;
  {
    unsigned int counter = 0;
    for (const auto &v : cgal_tria.finite_vertex_handles())
      {
        v_map[v] = counter++;
        vertices.emplace_back(CGALWrappers::to_dealii<2>(v->point()));
      }
  }
  for (const auto &cell : cgal_tria.finite_face_handles())
    {
      CellData<2> c(3);
      c.vertices[0] = v_map.at(cell->vertex(0));
      c.vertices[1] = v_map.at(cell->vertex(1));
      c.vertices[2] = v_map.at(cell->vertex(2));
      cells.emplace_back(c);
    }

  Triangulation<2> tria;
  tria.create_triangulation(vertices, cells, subcells);
  ASSERT_EQ(tria.n_active_cells(), cgal_tria.number_of_faces());
  ASSERT_EQ(tria.n_vertices(), cgal_tria.number_of_vertices());
}


TYPED_TEST(DimSpacedimTester, CGALConversions)
{
  constexpr auto dim      = TestFixture::dim;
  constexpr auto spacedim = TestFixture::spacedim;

  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 5, 6, 8}};
  // std::vector<std::vector<double>>       measures = {{},
  //                                              {1},
  //                                              {0.5, 1.0},
  //                                              {1. / 6., 4.
  //                                              / 3., 0.5,
  //                                              1}};

  // unsigned int i = 0;
  for (const auto nv : d2t[dim])
    {
      Triangulation<dim, spacedim> tria;
      Polyhedron                   poly;

      const auto ref     = ReferenceCell::n_vertices_to_type(dim, nv);
      const auto mapping = ref.template get_default_mapping<dim, spacedim>(1);

      GridGenerator::reference_cell(tria, ref);

      const auto cell = tria.begin_active();
      CGALWrappers::to_cgal(cell,
                            *mapping,
                            poly); // build CGAL Poly from deal.II cell
      ASSERT_TRUE(poly.is_valid() || dim == 1);
      if (dim == 3)
        {
          ASSERT_TRUE(poly.is_closed());
        }
    }
}


TEST(CGAL, SurfaceMesh)
{
  Triangulation<3> tria;
  const auto       ref     = ReferenceCell::n_vertices_to_type(3, 8);
  const auto       mapping = ref.template get_default_mapping<3, 3>(1);
  GridGenerator::reference_cell(tria, ref);

  typedef CGAL::Surface_mesh<K::Point_3> Mesh_surface;
  Mesh_surface                           surf;
  CGALWrappers::to_cgal(tria.begin_active(), *mapping, surf);
  ASSERT_TRUE(surf.is_valid());
}



TEST(CGAL, IntersectCubes)
{
  typedef CGAL::Surface_mesh<K::Point_3> Mesh_surface;
  namespace params = CGAL::Polygon_mesh_processing::parameters;

  Triangulation<3> tria;
  GridGenerator::hyper_cube(tria, -1., 1.);
  const auto &mapping = get_default_linear_mapping(tria);

  Mesh_surface surf1;
  assert(surf1.is_valid());

  CGALWrappers::to_cgal(tria.begin_active(), mapping, surf1);
  CGAL::Polygon_mesh_processing::triangulate_faces(surf1);



  GridTools::rotate(numbers::PI_4, 0, tria);
  // GridTools::rotate(numbers::PI_4, 2, tria);
  Mesh_surface surf2;
  assert(surf2.is_valid());
  CGALWrappers::to_cgal(tria.begin_active(), mapping, surf2);
  CGAL::Polygon_mesh_processing::triangulate_faces(surf2);


  std::array<boost::optional<Mesh_surface *>, 4> boolean_operations;

  Mesh_surface out_intersection, out_union;

  boolean_operations
    [CGAL::Polygon_mesh_processing::Corefinement::INTERSECTION] =
      &out_intersection;
  boolean_operations[CGAL::Polygon_mesh_processing::Corefinement::UNION] =
    &out_union;



  auto res =
    CGAL::Polygon_mesh_processing::corefine_and_compute_boolean_operations(
      surf1,
      surf2,
      boolean_operations,
      params::all_default(), // mesh1 named parameters
      params::all_default(), // mesh2 named parameters
      std::make_tuple(
        params::vertex_point_map(get(
          boost::vertex_point, out_union)), // named parameters for out_union
        params::vertex_point_map(
          get(boost::vertex_point,
              out_intersection)), // named parameters for out_intersection
        params::all_default(),    // named parameters for mesh1-mesh2 not used
        params::all_default())    // named parameters for mesh2-mesh1 not used)
    );
  ASSERT_TRUE(res[CGAL::Polygon_mesh_processing::Corefinement::INTERSECTION]);
}
