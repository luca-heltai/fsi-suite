#ifndef dealii_cgal_wrappers_h
#define dealii_cgal_wrappers_h

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#ifdef DEAL_II_WITH_CGAL

#  include <boost/config.hpp>

#  include <CGAL/Bbox_2.h>
#  include <CGAL/Bbox_3.h>
#  include <CGAL/Cartesian.h>
#  include <CGAL/IO/io.h>
#  include <CGAL/Origin.h>

namespace CGALWrappers
{
  template <typename CGALPointType, int dim>
  inline CGALPointType
  to_cgal(const dealii::Point<dim> &p)
  {
    if constexpr (dim == 1)
      return CGALPointType(p[0]);
    else if constexpr (dim == 2)
      return CGALPointType(p[0], p[1]);
    else if constexpr (dim == 3)
      return CGALPointType(p[0], p[1], p[2]);
    else
      Assert(false, dealii::ExcNotImplemented());
  }

  template <int dim, typename CGALPointType>
  inline dealii::Point<dim>
  to_dealii(const CGALPointType &p)
  {
    if constexpr (dim == 1)
      return dealii::Point<dim>(CGAL::to_double(p.x()));
    else if constexpr (dim == 2)
      return dealii::Point<dim>(CGAL::to_double(p.x()), CGAL::to_double(p.y()));
    else if constexpr (dim == 3)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                CGAL::to_double(p.y()),
                                CGAL::to_double(p.z()));
    else
      Assert(false, dealii::ExcNotImplemented());
  }
} // namespace CGALWrappers
#endif
#endif