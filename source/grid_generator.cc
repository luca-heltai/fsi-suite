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

#include "parsed_tools/grid_generator.h"

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>
#ifdef DEAL_II_WITH_OPENCASCADE
#  include <BRepBndLib.hxx>
#  include <Bnd_Box.hxx>
#  include <TopoDS.hxx>
#endif

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace dealii;

namespace ParsedTools
{
  template <int dim, int spacedim>
  GridGenerator<dim, spacedim>::GridGenerator(
    const std::string &prm_section_path,
    const std::string &grid_generator_function,
    const std::string &grid_generator_arguments,
    const std::string &output_file_name,
    const bool         transform_to_simplex_grid,
    const unsigned int initial_grid_refinement)
    : ParameterAcceptor(prm_section_path)
    , grid_generator_function(grid_generator_function)
    , grid_generator_arguments(grid_generator_arguments)
    , output_file_name(output_file_name)
    , transform_to_simplex_grid(transform_to_simplex_grid)
    , initial_grid_refinement(initial_grid_refinement)
  {
    add_parameter("Input name", this->grid_generator_function);
    add_parameter("Arguments", this->grid_generator_arguments);
    add_parameter("Output name", this->output_file_name);
    add_parameter("Transform to simplex grid", this->transform_to_simplex_grid);
    add_parameter("Initial grid refinement", this->initial_grid_refinement);
  }


  template <int dim, int spacedim>
  void
  GridGenerator<dim, spacedim>::generate(
    dealii::Triangulation<dim, spacedim> &tria) const
  {
    // TimerOutput::Scope timer_section(timer, "GridGenerator::generate");
    const auto ext =
      boost::algorithm::to_lower_copy(grid_generator_function.substr(
        grid_generator_function.find_last_of('.') + 1));

    // No extension was found: use grid generator functions
    if (ext == boost::algorithm::to_lower_copy(grid_generator_function) ||
        ext == "")
      {
        dealii::GridGenerator::generate_from_name_and_arguments(
          tria, grid_generator_function, grid_generator_arguments);
      }
    else
      {
        // grid_generator_function is a filename. Use GridIn
        GridIn<dim, spacedim> gi(tria);
        try
          {
            // Use gmsh api by default
            if (ext == "msh")
              gi.read_msh(grid_generator_function);
            // Otherwise try deal.II stdandard way of reading grids
            else
              gi.read(grid_generator_function); // Try default ways
          }
        catch (std::exception &exc)
          {
            std::cerr << std::endl
                      << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            std::cerr << "Exception on processing: " << std::endl
                      << exc.what() << std::endl
                      << "Trying other strategies." << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            // Attempt some of the other things manually
            std::ifstream in(grid_generator_function);
            AssertThrow(in, ExcIO());
            if (ext == "ar")
              {
                boost::archive::text_iarchive ia(in);
                tria.load(ia, 0);
              }
            else if (ext == "bin")
              {
                boost::archive::binary_iarchive ia(in);
                tria.load(ia, 0);
              }
            else
              {
                in.close();
                // try assimp reader as a last resort
                gi.read_assimp(grid_generator_function);
              }
          }

#ifdef DEAL_II_WITH_OPENCASCADE
        // Only if we read a file, we check if arguments can be interpreted as
        // manifold CAD files.
        using map_type  = std::map<types::manifold_id, std::string>;
        using Converter = Patterns::Tools::Convert<map_type>;
        for (const auto &[manifold_id, cad_file_name] :
             Converter::to_value(grid_generator_arguments))
          {
            const auto extension = boost::algorithm::to_lower_copy(
              cad_file_name.substr(cad_file_name.find_last_of('.') + 1));
            TopoDS_Shape shape;
            if (extension == "iges" || extension == "igs")
              shape = OpenCASCADE::read_IGES(cad_file_name, 1.0);
            else if (extension == "step" || extension == "stp")
              shape = OpenCASCADE::read_STEP(cad_file_name, 1.0);
            else if (extension == "stl")
              shape = OpenCASCADE::read_STL(cad_file_name);
            else
              AssertThrow(false,
                          ExcNotImplemented("We found an extension that we "
                                            "do not recognize as a CAD file "
                                            "extension. Bailing out."));
            if constexpr (spacedim > 1)
              {
                const auto [n_faces, n_edges, n_vertices] =
                  OpenCASCADE::count_elements(shape);
                deallog << "Manifold id: " << manifold_id << std::endl
                        << ", attached to CAD File " << cad_file_name
                        << std::endl;
                deallog.push(cad_file_name);
                deallog << "Number of faces: " << n_faces << std::endl
                        << "Number of edges: " << n_edges << std::endl
                        << "Number of vertices: " << n_vertices << std::endl;
                deallog.push("Bounding box");

                double Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;

                // Compute the parameters
                Bnd_Box box;
                BRepBndLib::Add(shape, box);
                box.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
                deallog << "Lower left corner: " << Xmin << ", " << Ymin << ", "
                        << Zmin << std::endl
                        << "Upper right corner: " << Xmax << ", " << Ymax
                        << ", " << Zmax << std::endl;
                deallog.pop();
                deallog.pop();
                if (n_faces == 0 && n_edges > 0)
                  tria.set_manifold(
                    manifold_id,
                    OpenCASCADE::ArclengthProjectionLineManifold<dim, spacedim>(
                      shape));
                else if constexpr (spacedim == 3)
                  {
                    tria.set_manifold(
                      manifold_id,
                      OpenCASCADE::NormalToMeshProjectionManifold<dim,
                                                                  spacedim>(
                        shape));
                  }
                else
                  tria.set_manifold(
                    manifold_id,
                    OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                      TopoDS::Face(shape)));
              }
          }
#else
        AssertThrow(grid_generator_arguments == "",
                    ExcNotImplemented("Generation of manifolds failed. We need "
                                      "OPENCASCADE to do this."));
#endif
      }

    // If the user wants to transform the grid to a simplex grid, do so
    if (transform_to_simplex_grid == true &&
        tria.all_reference_cells_are_hyper_cube() == true)
      {
        Triangulation<dim, spacedim> tmp;
        tmp.copy_triangulation(tria);
        tria.clear();
        dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tmp, tria);

        // Also copy all manifold ids
        for (const auto i : tmp.get_manifold_ids())
          if (i != numbers::flat_manifold_id)
            tria.set_manifold(i, tmp.get_manifold(i));
      }


    // Write the grid before refining it.
    write(tria);
    tria.refine_global(initial_grid_refinement);
  }



  template <int dim, int spacedim>
  void
  GridGenerator<dim, spacedim>::write(
    const dealii::Triangulation<dim, spacedim> &tria,
    const std::string                          &filename) const
  {
    const std::string outname = filename != "" ? filename : output_file_name;
    if (outname != "")
      {
        const auto ext = boost::algorithm::to_lower_copy(
          outname.substr(outname.find_last_of('.') + 1));

        GridOut go;
        if (ext == "msh")
          go.write_msh(tria, outname); // prefer msh api
        else
          {
            std::ofstream out(outname);
            AssertThrow(out, ExcIO());

            go.set_flags(GridOutFlags::Msh(true, true));
            go.set_flags(GridOutFlags::Ucd(false, true, true));
            go.set_flags(GridOutFlags::Vtu(true));

            if (ext == "vtk")
              go.write_vtk(tria, out);
            else if (ext == "vtu")
              go.write_vtu(tria, out);
            else if (ext == "ucd" || ext == "inp")
              go.write_ucd(tria, out);
            else if (ext == "vtu")
              go.write_vtu(tria, out);
            else if (ext == "ar")
              {
                boost::archive::text_oarchive oa(out);
                tria.save(oa, 0);
              }
            else if (ext == "bin")
              {
                boost::archive::binary_oarchive oa(out);
                tria.save(oa, 0);
              }
            else
              Assert(false, ExcNotImplemented());
            out.close();
          }
      }
  }


  template class GridGenerator<1, 1>;
  template class GridGenerator<1, 2>;
  template class GridGenerator<1, 3>;
  template class GridGenerator<2, 2>;
  template class GridGenerator<2, 3>;
  template class GridGenerator<3, 3>;

} // namespace ParsedTools
