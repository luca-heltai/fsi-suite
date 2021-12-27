#include <deal.II/base/patterns.h>

#include <deal.II/grid/reference_cell.h>

#include "tools/parsed_grid_generator.h"

using namespace dealii;

namespace dealii
{
  namespace Patterns
  {
    namespace Tools
    {
      template <>
      struct Convert<types::manifold_id>
      {
        using T = types::manifold_id;

        static std::unique_ptr<Patterns::PatternBase>
        to_pattern()
        {
          return Convert<int>::to_pattern();
        }

        static std::string
        to_string(
          const T &                    t,
          const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
        {
          return Convert<int>::to_string((int)(t), pattern);
        }

        static T
        to_value(
          const std::string &          s,
          const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
        {
          return T(Convert<int>::to_value(s, pattern));
        }
      };


      template <>
      struct Convert<ReferenceCell>
      {
        using T = ReferenceCell;

        static std::unique_ptr<Patterns::PatternBase>
        to_pattern()
        {
          return Convert<int>::to_pattern();
        }

        static std::string
        to_string(
          const T &                    t,
          const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
        {
          return Convert<int>::to_string(t, pattern);
        }

        static T
        to_value(
          const std::string &          s,
          const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
        {
          return dealii::internal::ReferenceCell::make_reference_cell_from_int(
            Convert<int>::to_value(s, pattern));
        }
      };
    } // namespace Tools
  }   // namespace Patterns
} // namespace dealii

namespace Tools
{
  template <int dim, int spacedim>
  struct GridInfo
  {
    GridInfo(const Triangulation<dim, spacedim> &tria,
             const unsigned int                  info_level = 0)
    {
      this->info_level = info_level;
      n_active_cells   = tria.n_active_cells();
      n_vertices       = tria.n_vertices();
      n_used_vertices  = tria.n_used_vertices();
      n_levels         = tria.n_levels();
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
        }
    }

    template <typename StreamType>
    void
    print_info(StreamType &out)
    {
      out << "Active cells  : " << n_active_cells << std::endl
          << "Vertices      : " << n_vertices << std::endl
          << "Used vertices : " << n_used_vertices << std::endl
          << "Levels        : " << n_levels << std::endl;
      if (info_level > 0)
        {
          out << "Active cells/level  : "
              << Patterns::Tools::to_string(n_active_cells_at_level)
              << std::endl
              << "Cells/level         : "
              << Patterns::Tools::to_string(n_cells_at_level) << std::endl;
        }
      if (info_level > 1)
        {
          out << "Boundary indicators : "
              << Patterns::Tools::to_string(boundary_ids) << std::endl
              << "Manifold ids         : "
              << Patterns::Tools::to_string(manifold_ids) << std::endl
              << "Reference cell types : "
              << Patterns::Tools::to_string(reference_cell_types) << std::endl;
        }
    }

    unsigned int info_level;
    unsigned int n_active_cells;
    unsigned int n_vertices;
    unsigned int n_used_vertices;
    unsigned int n_levels;

    std::vector<unsigned int> n_active_cells_at_level;
    std::vector<unsigned int> n_cells_at_level;

    std::vector<types::boundary_id> boundary_ids;
    std::vector<types::material_id> material_ids;
    std::vector<types::manifold_id> manifold_ids;
    std::vector<ReferenceCell>      reference_cell_types;
  };
} // namespace Tools

/**
 * Generate a mesh reader.
 */
int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      std::string                      par_name   = "";
      unsigned int                     info_level = 0;
      if (argc > 1)
        par_name = argv[1];
      if (argc > 2)
        info_level = std::atoi(argv[2]);


      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      Tools::ParsedGridGenerator<2> pgg;
      ParameterAcceptor::initialize(par_name);
      Triangulation<2> tria;
      pgg.generate(tria);
      Tools::GridInfo info(tria, info_level);
      info.print_info(deallog);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
