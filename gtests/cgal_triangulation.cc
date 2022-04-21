#include <deal.II/base/config.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

// #ifdef DEAL_II_WIHT_CGAL

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_2.h>

#include "cgal/wrappers.h"

// typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Simple_cartesian<double>    K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef Delaunay::Point                   DPoint;

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

// #endif