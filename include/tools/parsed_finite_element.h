#ifndef parsed_finite_element_h
#define parsed_finite_element_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/fe.h>

namespace Tools
{
  /**
   * Parsed FiniteElement. Read from a parameter file the name of a
   * finite element, generate one for you, and return a pointer to it.
   *
   * The name must be in the form which is returned by the
   * FiniteElement::get_name() function, where dimension template
   * parameters <2> etc. can be omitted. Alternatively, the explicit
   * number can be replaced by `dim` or `d`. If a number is given, it
   * must match the template parameter of this function.
   *
   * The names of FESystem elements follow the pattern
   * `FESystem[FE_Base1^p1-FE_Base2^p2]` The powers p1 etc. may either be
   * numbers or can be replaced by dim or d.
   *
   * If no finite element can be reconstructed from this string, an
   * exception of type FETools::ExcInvalidFEName is thrown.
   *
   * The operator() returns a pointer to a newly create finite element. It
   * is in the caller's responsibility to destroy the object pointed to
   * at an appropriate later time.
   *
   * Since the value of the template argument can't be deduced from the
   * (string) argument given to this function, you have to explicitly
   * specify it when you call this function.
   *
   * This function knows about all the standard elements defined in the
   * library. However, it doesn't by default know about elements that
   * you may have defined in your program. To make your own elements
   * known to this function, use the add_fe_name() function.
   */
  template <int dim, int spacedim = dim>
  class ParsedFiniteElement : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor. Takes a name for the section of the Parameter
     * Handler to use.
     *
     * This class is derived from ParameterAcceptor. See the documentation of
     * ParameterAcceptor for a guide on how section names and parameters are
     * parsed by this class.
     *
     * The optional parameters specify the component names as a comma separated
     * list of component names, and the default FiniteElement space name to use.
     * This class will throw an exception if the number of components does not
     * match the number of component names.
     *
     * The FiniteElement space name must be in the form which is returned by
     * the FiniteElement::get_name function, where dimension template
     * parameters <2> etc. can be omitted. Alternatively, the explicit
     * number can be replaced by dim or d. If a number is given, it must
     * match the template parameter of this function.
     *
     * The names of FESystem elements follow the pattern
     * FESystem[FE_Base1^p1-FE_Base2^p2] The powers p1 etc. may either be
     * numbers or can be replaced by dim or d.
     *
     * If no finite element can be reconstructed from this string, an
     * exception of type FETools::ExcInvalidFEName is thrown.
     *
     * Explicit conversion to the stored FiniteElement is possible after parsing
     * has occurred.
     *
     * If a component name is repeated, then that component is assumed to be
     * part of a vector or Tensor field, and it is treated as a single block.
     * User classes can use this information to construct block matrices and
     * vectors, or to group solution names according to components. For example,
     * a Stokes problem may have "u,u,p" for dim = 2 or "u, u, u, p" for dim
     * = 3, resulting in a vector-valued system with two blocks (one for the
     * velocity, and one for the pressure), and dim+1 components.
     */
    ParsedFiniteElement(const std::string &section_name    = "",
                        const std::string &component_names = "u",
                        const std::string &fe_name         = "FE_Q(1)");

    /**
     * Return a reference to the Finite Element.
     */
    operator dealii::FiniteElement<dim, spacedim> &();

    /**
     * Return a const reference to the Finite Element.
     */
    operator const dealii::FiniteElement<dim, spacedim> &() const;


    /**
     * Return a reference to the Finite Element.
     */
    dealii::FiniteElement<dim, spacedim> &
    operator()();

    /**
     * Return a const reference to the Finite Element.
     */
    const dealii::FiniteElement<dim, spacedim> &
    operator()() const;

    /**
     * Return the component names for this Finite Element.
     */
    const std::string &
    get_joint_component_names() const;


    /**
     * Return the component names for this Finite Element.
     */
    const std::vector<std::string> &
    get_component_names() const;

    /**
     * Return the blocking of the components for this finite
     * element. This is what's needed by the block renumbering
     * algorithm.
     */
    std::vector<unsigned int>
    get_component_blocks() const;


    /**
     * Return the block names for this Finite Element.
     */
    std::string
    get_block_names() const;

    /**
     * Return the number of components of the Finite Element.
     */
    unsigned int
    n_components() const;

    /**
     * Return the number of blocks of the Finite Element, i.e.,
     * the number of variables. For example, simple Heat equation
     * has 1 block, Navier-Stokes 2 blocks (u and p).
     */
    unsigned int
    n_blocks() const;

    /**
     * Return the first occurence of @p var in @p default_component_names.
     * Remark: this is the value required by FEValuesExtractors.
     */
    unsigned int
    get_first_occurence(const std::string &var) const;

    /**
     * Return @p true if @p var is vector, @p false otherwise.
     */
    bool
    is_vector(const std::string &var) const;

  protected:
    /**
     * Component names as a single comma separated string.
     */
    const std::string joint_component_names;

    /**
     * Component names. This is comma separeted list of component names
     * which identifies the Finite Elemenet. If a name is repeated, then
     * that component is assumed to be part of a vector field or Tensor
     * field, and is treated as a single block. User classes can use
     * this information to construct block matrices and vectors, or to
     * group solution names according to components. For example, a
     * Stokes problem may have "u, u, p" for dim = 2 or "u, u, u, p" for
     * dim = 3.
     */
    const std::vector<std::string> component_names;

    /**
     * Finite Element Name.
     */
    std::string fe_name;

    /**
     * The subdivision, in terms of component indices. This is
     * automatically computed from the the component names.
     */
    std::vector<unsigned int> component_blocks;

    /**
     * The subdivision, in terms of block names. This is automatically
     * computed from the the component names.
     */
    std::vector<std::string> block_names;

    /** The actual FiniteElement object. */
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe;
  };

} // namespace Tools

#endif
