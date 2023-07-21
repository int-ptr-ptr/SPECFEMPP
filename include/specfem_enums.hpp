#ifndef SPECFEM_ENUM_HPP
#define SPECFEM_ENUM_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
/**
 * @brief enums namespace is used to store enumerations.
 *
 */
namespace enums {

/**
 * @brief Cartesian axes
 *
 */
enum class axes {
  x, ///< X axis
  y, ///< Y axis
  z  ///< Z axis
};

namespace seismogram {
enum class type {
  displacement, ///< Displacement seismogram
  velocity,     ///< Velocity Seismogram
  acceleration  ///< Acceleration seismogram
};

std::ostream &operator<<(std::ostream &os, const type &t) {
  switch (t) {
  case type::displacement:
    os << "displacement";
    break;
  case type::velocity:
    os << "velocity";
    break;
  case type::acceleration:
    os << "acceleration";
    break;
  default:
    os << "unknown";
    break;
  }
  return os;
}

enum format {
  seismic_unix, ///< Seismic unix output format
  ascii         ///< ASCII output format
};

} // namespace seismogram

/**
 * @brief element namespace is used to store element properties used in the
 * element class.
 *
 */
namespace element {

/**
 * @brief type of element
 *
 * This is primarily used to label the element as elastic, acoustic or
 * poroelastic.
 *
 */
enum type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};

/**
 * @brief dimensionality property of the element
 *
 */
namespace dimension {
/**
 * @brief 2D element
 *
 */
class dim2 {
public:
  constexpr static int dim = 2;
};
/**
 * @brief 3D element
 *
 */
class dim3 {
public:
  constexpr static int dim = 3;
};
} // namespace dimension

/**
 * @brief number of quadrature points defined either at compile time or at
 * runtime
 *
 */
namespace quadrature {

/**
 * @brief define the number of quadrature points at runtime
 *
 */
class dynamic_quadrature_points {
public:
  int ngllx; ///< number of quadrature points in the x direction
  int ngllz; ///< number of quadrature points in the z direction

  /**
   * @brief Scratch memory space. Shared memory for GPUs and thread local memory
   * for CPUs.
   *
   */
  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  /**
   * @brief Team member type. See Kokkos TeamHandle documentation for more
   * details.
   *
   */
  using member_type = specfem::kokkos::DeviceTeam::member_type;

  /**
   * @brief Scratch view type
   *
   * For dynamic quadrature points, the scratch view type is a 2D dynamic view
   * stored on scratch space.
   *
   * @tparam T
   */
  template <typename T>
  using ScratchViewType = specfem::kokkos::DeviceScratchView2d<T>;

  /**
   * @brief Deleted default constructor
   *
   */
  dynamic_quadrature_points() = delete;

  /**
   * @brief Construct a new dynamic quadrature points object
   *
   * @param ngllz Number of quadrature points in the z direction
   * @param ngllx Number of quadrature points in the x direction
   */
  dynamic_quadrature_points(const int &ngllz, const int &ngllx)
      : ngllx(ngllx), ngllz(ngllz){};

  /**
   * @brief Destroy the dynamic quadrature points object
   *
   */
  ~dynamic_quadrature_points() = default;

  /**
   * @brief Get the required size of the scratch memory space for a given type T
   * and axes ax1 and ax2
   *
   * @tparam T Tyoe of the scratch view
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @return std::size_t Size of the scratch memory space
   */
  template <typename T, specfem::enums::axes ax1, specfem::enums::axes ax2>
  std::size_t shmem_size() const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T>::shmem_size(this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T>::shmem_size(this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T>::shmem_size(this->ngllz, this->ngllx);
    }
  }

  /**
   * @brief Get the scratch view for a given type T and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @param ptr Address in the scratch memory space to allocate the scratch view
   * @return ScratchViewType<T> Scratch view allocated at the address ptr
   */
  template <typename T, specfem::enums::axes ax1, specfem::enums::axes ax2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T>
  ScratchView(scratch_memory_space &ptr) const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return ScratchViewType<T>(ptr, this->ngllx, this->ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return ScratchViewType<T>(ptr, this->ngllz, this->ngllz);
    } else {
      return ScratchViewType<T>(ptr, this->ngllz, this->ngllx);
    }
  };

  /**
   * @brief Get the team thread range for a given axes ax1 and ax2
   *
   * @tparam ax1 Axis 1
   * @tparam ax2 Axis 2
   * @param team_member Team member
   * @return Kokkos::TeamThreadRange Team thread range
   */
  template <specfem::enums::axes ax1, specfem::enums::axes ax2>
  KOKKOS_INLINE_FUNCTION auto
  TeamThreadRange(const member_type &team_member) const {
    if constexpr (ax1 == specfem::enums::axes::x &&
                  ax2 == specfem::enums::axes::x) {
      return Kokkos::TeamThreadRange(team_member, ngllx * ngllx);
    } else if constexpr (ax1 == specfem::enums::axes::z &&
                         ax2 == specfem::enums::axes::z) {
      return Kokkos::TeamThreadRange(team_member, ngllz * ngllz);
    } else {
      return Kokkos::TeamThreadRange(team_member, ngllz * ngllx);
    }
  }

  /**
   * @brief Get the number of quadrature points in the x and z directions
   *
   * @param ngllx Number of quadrature points in the x direction
   * @param ngllz Number of quadrature points in the z direction
   */
  KOKKOS_INLINE_FUNCTION void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = this->ngllx;
    *ngllz = this->ngllz;
  }
};

// Define the number of quadrature points at compile time
template <int N> class static_quadrature_points {

public:
  constexpr static int NGLL = N; ///< Number of quadrature points

  /**
   * @brief Scratch memory space type
   *
   */
  using scratch_memory_space =
      specfem::kokkos::DevExecSpace::scratch_memory_space;

  /**
   * @brief Team member type. See kokkos TeamHandle documentation for more
   * details.
   *
   */
  using member_type = specfem::kokkos::DeviceTeam::member_type;

  /**
   * @brief Scratch view type for a given type T.
   *
   * For static quadrature points, the scratch view is a 2D view of size NGLL x
   * NGLL defined at compile time.
   *
   * @tparam T Type of the scratch view
   */
  template <typename T>
  using ScratchViewType =
      specfem::kokkos::StaticDeviceScratchView2d<T, NGLL, NGLL>;

  /**
   * @brief Construct a new static quadrature points object
   *
   */
  constexpr static_quadrature_points() = default;
  /**
   * @brief Destroy the static quadrature points object
   *
   */
  ~static_quadrature_points() = default;

  /**
   * @brief Get the required size of the scratch memory space for a given type T
   * and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @return std::size_t Size of the scratch memory space
   */
  template <typename T, specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  std::size_t shmem_size() const {
    return ScratchViewType<T>::shmem_size();
  }

  /**
   * @brief Get the scratch view for a given type T and axes ax1 and ax2
   *
   * @tparam T Type of the scratch view
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @param ptr Address in the scratch memory space to allocate the scratch view
   * @return ScratchViewType<T> Scratch view allocated at the address ptr
   */
  template <typename T, specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION ScratchViewType<T>
  ScratchView(const scratch_memory_space &ptr) const {
    return ScratchViewType<T>(ptr);
  }

  /**
   * @brief Get the team thread range for a given axes ax1 and ax2
   *
   * @tparam ax_1 Axis 1
   * @tparam ax_2 Axis 2
   * @param team_member Team member
   * @return Kokkos::TeamThreadRange Team thread range
   */
  template <specfem::enums::axes ax_1, specfem::enums::axes ax_2>
  KOKKOS_INLINE_FUNCTION auto
  TeamThreadRange(const member_type &team_member) const {
    return Kokkos::TeamThreadRange(team_member, NGLL * NGLL);
  }

  /**
   * @brief Get the number of quadrature points in the x and z directions
   *
   * @param ngllx Number of quadrature points in the x direction
   * @param ngllz Number of quadrature points in the z direction
   */
  KOKKOS_INLINE_FUNCTION constexpr void get_ngll(int *ngllx, int *ngllz) const {
    *ngllx = NGLL;
    *ngllz = NGLL;
  }
};

} // namespace quadrature

/**
 * @brief medium property of the element
 *
 */
namespace medium {
/**
 * @brief Elastic medium
 *
 */
class elastic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::elastic;
  /**
   * @brief constexpr defining number of components for this medium.
   *
   */
  constexpr static int components = 2;
};

/**
 * @brief Acoustic medium
 *
 */
class acoustic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::acoustic;
  /**
   * @brief constexpr defining number of components for this medium.
   *
   */
  constexpr static int components = 1;
};

} // namespace medium

/**
 * @brief Elemental properties
 *
 * Properties can be utilized to distinguish elements based on physics or to
 * optimize kernel calculations for specific elements.
 */
namespace property {
class isotropic {};
} // namespace property
} // namespace element
} // namespace enums
} // namespace specfem

#endif
