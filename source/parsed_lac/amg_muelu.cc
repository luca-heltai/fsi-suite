#include <parsed_lac/amg_muelu.h>

#if defined(DEAL_II_WITH_TRILINOS) && defined(DEAL_II_TRILINOS_WITH_MUELU)

#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/lac/sparse_matrix.h>

using namespace dealii;

namespace ParsedLAC
{
  AMGMueLuPreconditioner::AMGMueLuPreconditioner(
    const std::string & name,
    const bool &        elliptic,
    const unsigned int &n_cycles,
    const bool &        w_cycle,
    const double &      aggregation_threshold,
    const unsigned int &smoother_sweeps,
    const unsigned int &smoother_overlap,
    const bool &        output_details,
    const std::string & smoother_type,
    const std::string & coarse_type)
    : ParameterAcceptor(name)
    , PreconditionAMGMueLu()
    , elliptic(elliptic)
    , n_cycles(n_cycles)
    , w_cycle(w_cycle)
    , aggregation_threshold(aggregation_threshold)
    , smoother_sweeps(smoother_sweeps)
    , smoother_overlap(smoother_overlap)
    , output_details(output_details)
    , smoother_type(smoother_type)
    , coarse_type(coarse_type)
  {
    add_parameters();
  }

  void
  AMGMueLuPreconditioner::add_parameters()
  {
    add_parameter(
      "Elliptic",
      elliptic,
      "Determines whether the AMG preconditioner should be optimized for "
      "elliptic problems (ML option smoothed aggregation SA, using a "
      "Chebyshev smoother) or for non-elliptic problems (ML option "
      "non-symmetric smoothed aggregation NSSA, smoother is SSOR with "
      "underrelaxation");

    add_parameter(
      "Number of cycles",
      n_cycles,
      "Defines how many multigrid cycles should be performed by the "
      "preconditioner.");

    add_parameter(
      "w-cycle",
      w_cycle,
      "defines whether a w-cycle should be used instead of the standard "
      "setting of a v-cycle.");

    add_parameter(
      "Aggregation threshold",
      aggregation_threshold,
      "This threshold tells the AMG setup how the coarsening should be "
      "performed. In the AMG used by ML, all points that strongly couple with "
      "the tentative coarse-level point form one aggregate. The term strong "
      "coupling is controlled by the variable aggregation_threshold, meaning "
      "that all elements that are not smaller than aggregation_threshold "
      "times the diagonal element do couple strongly.");

    add_parameter(
      "Smoother sweeps",
      smoother_sweeps,
      "Determines how many sweeps of the smoother should be performed. When "
      "the flag elliptic is set to true, i.e., for elliptic or almost "
      "elliptic problems, the polynomial degree of the Chebyshev smoother is "
      "set to smoother_sweeps. The term sweeps refers to the number of "
      "matrix-vector products performed in the Chebyshev case. In the "
      "non-elliptic case, smoother_sweeps sets the number of SSOR relaxation "
      "sweeps for post-smoothing to be performed.");

    add_parameter(
      "Smoother overlap",
      smoother_overlap,
      "Determines the overlap in the SSOR/Chebyshev error smoother when run "
      "in parallel.");

    add_parameter(
      "Output details",
      output_details,
      "If this flag is set to true, then internal information from the ML "
      "preconditioner is printed to screen. This can be useful when debugging "
      "the preconditioner.");

    add_parameter(
      "Smoother type",
      smoother_type,
      "Determines which smoother to use for the AMG cycle.",
      this->prm,
      Patterns::Selection(
        "|Aztec|IFPACK|Jacobi"
        "|ML symmetric Gauss-Seidel|symmetric Gauss-Seidel"
        "|ML Gauss-Seidel|Gauss-Seidel|block Gauss-Seidel"
        "|symmetric block Gauss-Seidel|Chebyshev|MLS|Hiptmair"
        "|Amesos-KLU|Amesos-Superlu|Amesos-UMFPACK|Amesos-Superludist"
        "|Amesos-MUMPS|user-defined|SuperLU|IFPACK-Chebyshev|self"
        "|do-nothing|IC|ICT|ILU|ILUT|Block Chebyshev"
        "|IFPACK-Block Chebyshev"));

    add_parameter(
      "Coarse type",
      coarse_type,
      "Determines which solver to use on the coarsest level. The same "
      "settings as for the smoother type are possible.",
      this->prm,
      Patterns::Selection(
        "|Aztec|IFPACK|Jacobi"
        "|ML symmetric Gauss-Seidel|symmetric Gauss-Seidel"
        "|ML Gauss-Seidel|Gauss-Seidel|block Gauss-Seidel"
        "|symmetric block Gauss-Seidel|Chebyshev|MLS|Hiptmair"
        "|Amesos-KLU|Amesos-Superlu|Amesos-UMFPACK|Amesos-Superludist"
        "|Amesos-MUMPS|user-defined|SuperLU|IFPACK-Chebyshev|self"
        "|do-nothing|IC|ICT|ILU|ILUT|Block Chebyshev"
        "|IFPACK-Block Chebyshev"));
  }

  template <typename Matrix>
  void
  AMGMueLuPreconditioner::initialize_preconditioner(const Matrix &matrix)
  {
    TrilinosWrappers::PreconditionAMGMueLu::AdditionalData data;

    data.elliptic              = elliptic;
    data.n_cycles              = n_cycles;
    data.w_cycle               = w_cycle;
    data.aggregation_threshold = aggregation_threshold;
    data.constant_modes        = std::vector<std::vector<bool>>(0);
    data.smoother_sweeps       = smoother_sweeps;
    data.smoother_overlap      = smoother_overlap;
    data.output_details        = output_details;
    data.smoother_type         = smoother_type.c_str();
    data.coarse_type           = coarse_type.c_str();
    this->initialize(matrix, data);
  }
} // namespace ParsedLAC

template void
ParsedLAC::AMGMueLuPreconditioner::initialize_preconditioner<
  dealii::TrilinosWrappers::SparseMatrix>(
  const dealii::TrilinosWrappers::SparseMatrix &);

template void
ParsedLAC::AMGMueLuPreconditioner::initialize_preconditioner<
  dealii::SparseMatrix<double>>(const dealii::SparseMatrix<double> &);

#endif
