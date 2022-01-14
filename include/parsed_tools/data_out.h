#ifndef parsed_tools_data_out_h
#define parsed_tools_data_out_h

#include <deal.II/base/config.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <filesystem>
#include <fstream>

#include "parsed_tools/enum.h"

namespace ParsedTools
{
  template <int dim, int spacedim = dim>
  class DataOut : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @brief Construct a new DataOut object
     *
     * @param name
     * @param base_name
     * @param output_format
     * @param subdivisions
     * @param output_high_order
     * @param comm
     */
    DataOut(const std::string & section_name             = "",
            const std::string & base_name                = "solution",
            const std::string & output_format            = "vtu",
            const unsigned int &subdivisions             = 0,
            const bool &        write_higher_order_cells = true,
            const MPI_Comm &    comm                     = MPI_COMM_WORLD);

    /**
     * Prepare to output data on the given file. This will initialize
     * the data_out object and a file with a filename that is the
     * combination of the @p base_name, the optional @p suffix,
     * eventually a processor number and the output suffix.
     */
    void
    attach_dof_handler(const dealii::DoFHandler<dim, spacedim> &dh,
                       const std::string &                      suffix = "");

    /**
     * Add the given vector to the output file. Prior to calling this
     * method, you have to call the attach_dof_handler() method. The
     * string can be a comma separated list of components, or a single
     * description. In this latter case, a progressive number per
     * component is added in the end. In the former case, if a string is
     * reapeated, its respective components are interepreted as components of a
     * vector.
     */
    template <typename VECTOR>
    void
    add_data_vector(const VECTOR &data_vector, const std::string &desc);

    /**
     * Wrapper for the corrisponding function in dealii.
     */
    template <typename VECTOR>
    void
    add_data_vector(const VECTOR &                             data_vector,
                    const dealii::DataPostprocessor<spacedim> &postproc);


    /**
     * Actually write the file. Once the data_out has been prepared,
     * vectors have been added, the data can be written to a file. This
     * is done in this method. At the end of this function call the
     * process can be started again.
     * @p used_files is an optional variable that takes a list of useful files
     * (ex. "parameter.prm time.dat") and copies these files
     * in the @p incremental_run_prefix of the costructor function.
     */
    void
    write_data_and_clear(const dealii::Mapping<dim, spacedim> &mapping =
                           dealii::StaticMappingQ1<dim, spacedim>::mapping);

    /** Resets the pvd_record. */
    void
    clear_pvd_record();

  private:
    /** Initialization flag.*/
    const std::string component_names;

    /** MPI communicator. */
    const MPI_Comm &comm;

    /** Number of processes. */
    const unsigned int n_mpi_processes;

    /** My mpi process. */
    const unsigned int this_mpi_process;

    /** Output format. */
    std::string output_format;

    /** Number of subdivisions. */
    unsigned int subdivisions;

    /** If available, output high order data. */
    bool write_higher_order_cells;

    /** Base name for output files. This base is used to generate all
        filenames. */
    std::string base_name;

    /** Folder where solutions are stored. */
    std::string current_directory;

    /** The name of last written filew. */
    std::string current_filename;

    /** The name used for the master pvd file. */
    std::string master_name;

    /** Current output name. When preparing data_out, this name will
        contain the base for the current output. This allows the user to
        use a different output name in different part of the program. */
    std::string current_name;

    /** Output the partitioning of the domain. */
    bool output_partitioning;

    /** Output the material ids of the domain. */
    bool output_material_ids;

    /** Output file. */
    std::ofstream output_file;

    /** Outputs only the data that refers to this process. */
    std::unique_ptr<dealii::DataOut<dim, spacedim>> data_out;

    typename dealii::DataOut<dim, spacedim>::CurvedCellRegion
      curved_cells_region = dealii::DataOut<dim, spacedim>::curved_inner_cells;

    /**
     * Record of all output files and times.
     */
    std::vector<std::pair<double, std::string>> pvd_record;
  };


  // ============================================================
  // Template specializations
  // ============================================================

  template <int dim, int spacedim>
  template <typename VECTOR>
  void
  DataOut<dim, spacedim>::add_data_vector(const VECTOR &     data_vector,
                                          const std::string &desc)
  {
    std::vector<std::string> dd = dealii::Utilities::split_string_list(desc);
    if (data_out->default_suffix() != "")
      {
        if (dd.size() == 1)
          {
            data_out->add_data_vector(data_vector, desc);
          }
        else
          {
            std::vector<std::string>::iterator sit = dd.begin();
            std::vector<int>                   occurrances;

            for (; sit != dd.end(); ++sit)
              occurrances.push_back(std::count(dd.begin(), dd.end(), *sit));

            std::vector<int>::iterator iit = occurrances.begin();
            sit                            = dd.begin();

            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation;

            for (; iit != occurrances.end(); ++iit, ++sit)
              {
                if (*iit > 1)
                  data_component_interpretation.push_back(
                    dealii::DataComponentInterpretation::
                      component_is_part_of_vector);
                else
                  data_component_interpretation.push_back(
                    dealii::DataComponentInterpretation::component_is_scalar);
              }

            data_out->add_data_vector(
              data_vector,
              dd,
              dealii::DataOut<dim, spacedim>::type_dof_data,
              data_component_interpretation);
          }
      }
  }


  template <int dim, int spacedim>
  template <typename VECTOR>
  void
  DataOut<dim, spacedim>::add_data_vector(
    const VECTOR &                             data_vector,
    const dealii::DataPostprocessor<spacedim> &postproc)
  {
    data_out->add_data_vector(data_vector, postproc);
  }

} // namespace ParsedTools

#endif
