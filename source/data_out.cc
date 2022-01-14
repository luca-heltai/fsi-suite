#include "parsed_tools/data_out.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

namespace
{
  // Return relative filename
  std::string
  relative(const std::string &filename)
  {
    auto rel = filename;
    if (rel.find_last_of('/') != std::string::npos)
      rel = rel.substr(rel.find_last_of('/') + 1);
    return rel;
  }

  // Return directory name
  std::string
  dirname(const std::string &filename)
  {
    auto rel = filename;
    if (rel.find_last_of('/') != std::string::npos)
      rel = rel.substr(0, rel.find_last_of('/'));
    else
      rel = "./";
    // Check if it exists, if not, create it
    AssertThrow(std::system(
                  ("test -d " + rel + " || mkdir -p " + rel).c_str()) == 0,
                ExcMessage("Could not create directory " + rel));
    return rel;
  }
} // namespace

namespace ParsedTools
{
  template <int dim, int spacedim>
  DataOut<dim, spacedim>::DataOut(const std::string & section_name,
                                  const std::string & base_name,
                                  const std::string & output_format,
                                  const unsigned int &subdivisions,
                                  const bool &        write_higher_order_cells,
                                  const MPI_Comm &    comm)
    : ParameterAcceptor(section_name)
    , comm(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , output_format(output_format)
    , subdivisions(subdivisions)
    , write_higher_order_cells(write_higher_order_cells)
    , base_name(base_name)
  {
    add_parameter("Problem base name", this->base_name);

    add_parameter("Output partitioning", this->output_partitioning);

    add_parameter("Output material ids", this->output_material_ids);

    add_parameter("Output format",
                  this->output_format,
                  "",
                  this->prm,
                  Patterns::Selection(DataOutBase::get_output_format_names()));

    add_parameter("Subdivisions", this->subdivisions);

    add_parameter("Write high order cells", this->write_higher_order_cells);

    add_parameter("Curved cells region", this->curved_cells_region);
  }



  template <int dim, int spacedim>
  void
  DataOut<dim, spacedim>::clear_pvd_record()
  {
    pvd_record.clear();
  }



  template <int dim, int spacedim>
  void
  DataOut<dim, spacedim>::attach_dof_handler(
    const DoFHandler<dim, spacedim> &dh,
    const std::string &              suffix)
  {
    data_out = std::make_unique<dealii::DataOut<dim, spacedim>>();
    data_out->set_default_format(
      DataOutBase::parse_output_format(output_format));


    master_name       = relative(base_name);
    auto fname        = base_name + (suffix.empty() ? "" : "_" + suffix);
    current_filename  = relative(fname);
    current_directory = dirname(fname);

    if (subdivisions == 0)
      subdivisions = dh.get_fe().degree;

    if (write_higher_order_cells && dim > 1)
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out->set_flags(flags);
      }

    if (data_out->default_suffix() != "")
      {
        // If the output is needed and we have many processes, just output
        // the one we need *in intermediate format*.
        if (n_mpi_processes > 1)
          fname += ("." + Utilities::int_to_string(this_mpi_process, 2) + "." +
                    Utilities::int_to_string(n_mpi_processes, 2) +
                    data_out->default_suffix());
        else
          fname += data_out->default_suffix();

        output_file.open(fname);
        AssertThrow(output_file, ExcIO());
        data_out->attach_dof_handler(dh);

        if (n_mpi_processes > 1)
          {
            // Output the partitioning
            if (output_partitioning)
              {
                Vector<float> partitioning(
                  dh.get_triangulation().n_active_cells());
                for (unsigned int i = 0; i < partitioning.size(); ++i)
                  partitioning(i) = this_mpi_process;
                static Vector<float> static_partitioning;
                static_partitioning.swap(partitioning);
                data_out->add_data_vector(static_partitioning, "partitioning");
              }
          }
        // Output the materialids
        if (output_material_ids)
          {
            static Vector<float> material_ids;
            material_ids.reinit(dh.get_triangulation().n_active_cells());
            for (const auto cell : dh.active_cell_iterators())
              material_ids(cell->active_cell_index()) = cell->material_id();
            data_out->add_data_vector(material_ids, "material_id");
          }
      }
  }



  template <int dim, int spacedim>
  void
  DataOut<dim, spacedim>::write_data_and_clear(
    const Mapping<dim, spacedim> &mapping)
  {
    if (output_format == "none")
      return;

    AssertThrow(output_file, ExcIO());
    if (data_out->default_suffix() != "")
      {
        data_out->build_patches(mapping,
                                this->subdivisions,
                                this->curved_cells_region);
        data_out->write(output_file);

        std::string master_file = current_filename;

        if (this_mpi_process == 0 && n_mpi_processes > 1 &&
            data_out->default_suffix() == ".vtu")
          {
            std::vector<std::string> filenames;
            for (unsigned int i = 0; i < n_mpi_processes; ++i)
              filenames.push_back(relative(current_filename) + "." +
                                  Utilities::int_to_string(i, 2) + "." +
                                  Utilities::int_to_string(n_mpi_processes, 2) +
                                  data_out->default_suffix());

            std::ofstream master_output(
              (relative(current_filename) + ".pvtu").c_str());
            data_out->write_pvtu_record(master_output, filenames);

            master_file += ".pvtu";
          }
        else
          {
            master_file += data_out->default_suffix();
          }

        pvd_record.push_back(
          std::make_pair(pvd_record.size(), relative(master_file)));

        if (this_mpi_process == 0)
          {
            std::ofstream pvd_output(
              (current_directory + "/" + master_name + ".pvd"));
            DataOutBase::write_pvd_record(pvd_output, pvd_record);
          }
      }
    data_out = 0;
    output_file.close();
  }

  template class DataOut<1, 1>;
  template class DataOut<1, 2>;
  template class DataOut<1, 3>;
  template class DataOut<2, 2>;
  template class DataOut<2, 3>;
  template class DataOut<3, 3>;
} // namespace ParsedTools
