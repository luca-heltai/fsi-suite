#ifndef amg_muelu_preconditioner_h
#define amg_muelu_preconditioner_h

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#if defined(DEAL_II_WITH_TRILINOS) && defined(DEAL_II_TRILINOS_WITH_MUELU)

#  include <deal.II/lac/trilinos_precondition.h>

namespace ParsedLAC
{
  /**
   * A parsed AMG preconditioner based on MueLu which uses parameter files to
   * choose between different options. This object is a
   * TrilinosWrappers::PreconditionAMGMueLu which can be called in place of
   * the preconditioner.
   */
  class AMGMueLuPreconditioner
    : public dealii::ParameterAcceptor,
      public dealii::TrilinosWrappers::PreconditionAMGMueLu
  {
  public:
    /**
     * Constructor. Build the preconditioner of a matrix using AMG.
     */
    AMGMueLuPreconditioner(const std::string & name                  = "",
                           const bool &        elliptic              = true,
                           const unsigned int &n_cycles              = 1,
                           const bool &        w_cycle               = false,
                           const double &      aggregation_threshold = 1e-4,
                           const unsigned int &smoother_sweeps       = 2,
                           const unsigned int &smoother_overlap      = 0,
                           const bool &        output_details        = false,
                           const std::string & smoother_type = "Chebyshev",
                           const std::string & coarse_type   = "Amesos-KLU");

    /**
     * Initialize the preconditioner using @p matrix.
     */
    template <typename Matrix>
    void
    initialize_preconditioner(const Matrix &matrix);

    using dealii::TrilinosWrappers::PreconditionAMGMueLu::initialize;

  private:
    /**
     * Add all parameter options.
     */
    void
    add_parameters();

    /**
     * Determines whether the AMG preconditioner should be optimized for
     * elliptic problems (ML option smoothed aggregation SA, using a
     * Chebyshev smoother) or for non-elliptic problems (ML option non-
     * symmetric smoothed aggregation NSSA, smoother is SSOR with
     * underrelaxation).
     */
    bool elliptic;

    /**
     * Defines how many multigrid cycles should be performed by the
     * preconditioner.
     */
    unsigned int n_cycles;

    /**
     * Defines whether a w-cycle should be used instead of the standard
     * setting of a v-cycle.
     */
    bool w_cycle;

    /**
     * This threshold tells the AMG setup how the coarsening should be
     * performed. In the AMG used by ML, all points that strongly couple
     * with the tentative coarse-level point form one aggregate. The
     * term strong coupling is controlled by the variable
     * aggregation_threshold, meaning that all elements that are not
     * smaller than aggregation_threshold times the diagonal element do
     * couple strongly.
     */
    double aggregation_threshold;

    /**
     * Determines how many sweeps of the smoother should be
     * performed. When the flag elliptic is set to true, i.e., for
     * elliptic or almost elliptic problems, the polynomial degree of
     * the Chebyshev smoother is set to smoother_sweeps. The term sweeps
     * refers to the number of matrix-vector products performed in the
     * Chebyshev case. In the non-elliptic case, smoother_sweeps sets
     * the number of SSOR relaxation sweeps for post-smoothing to be
     * performed.
     */
    unsigned int smoother_sweeps;

    /**
     * Determines the overlap in the SSOR/Chebyshev error smoother when
     * run in parallel.
     */
    unsigned int smoother_overlap;

    /**
     * If this flag is set to true, then internal information from the
     * ML preconditioner is printed to screen. This can be useful when
     * debugging the preconditioner.
     */
    bool output_details;

    /**
     * Determines which smoother to use for the AMG cycle. Possibilities
     * for smoother_type are the following: "Aztec", "IFPACK", "Jacobi",
     * "ML symmetric Gauss-Seidel", "symmetric Gauss-Seidel", "ML
     * Gauss-Seidel", "Gauss-Seidel", "block Gauss-Seidel", "symmetric
     * block Gauss-Seidel", "Chebyshev", "MLS", "Hiptmair",
     * "Amesos-KLU", "Amesos-Superlu", "Amesos-UMFPACK",
     * "Amesos-Superludist", "Amesos-MUMPS", "user-defined", "SuperLU",
     * "IFPACK-Chebyshev", "self", "do-nothing", "IC", "ICT", "ILU",
     * "ILUT", "Block Chebyshev", "IFPACK-Block Chebyshev"
     */
    std::string smoother_type;

    /**
     * Determines which solver to use on the coarsest level. The same
     * settings as for the smoother type are possible.
     */
    std::string coarse_type;
  };


} // namespace ParsedLAC

#endif // DEAL_II_WITH_TRILINOS

#endif
