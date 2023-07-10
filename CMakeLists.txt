CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(gtest)

FIND_PACKAGE(deal.II 9.3 REQUIRED
HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Fix warning on mac
SET(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set preference between CGAL and MOONOLITH for intersections
option(PREFER_CGAL_OVER_PARMOONOLITH "If both CGAL and PARMOONOLITH are found, use CGAL in compute_intersections." ON)
ADD_COMPILE_DEFINITIONS(DEAL_II_PREFER_CGAL_OVER_PARMOONOLITH)

# # CGAL 
# set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE "TRUE")
# find_package(CGAL)
# if(CGAL_FOUND)
#     ADD_COMPILE_DEFINITIONS(DEAL_II_WITH_CGAL)
#     if(PREFER_CGAL_OVER_PARMOONOLITH)
#         ADD_COMPILE_DEFINITIONS(DEAL_II_PREFER_CGAL_OVER_PARMOONOLITH)
#     endif()
# MESSAGE("-- DEAL_II_WITH_CGAL = ON")
# else()
# MESSAGE("-- DEAL_II_WITH_CGAL = OFF")
# ENDIF(CGAL_FOUND)

# ParMoonolith
find_package(ParMoonolith HINTS /usr/local/lib/cmake)
if(ParMoonolith_FOUND)
ADD_COMPILE_DEFINITIONS(DEAL_II_WITH_PARMOONOLITH)
ADD_COMPILE_DEFINITIONS(FSI_SUITE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE("-- DEAL_II_WITH_PARMOONOLITH = ON")
else()
MESSAGE("-- DEAL_II_WITH_PARMOONOLITH = OFF")
ENDIF(ParMoonolith_FOUND)

# Gather files, libraries, and apps
file(GLOB LIB_SOURCES source/*.cc source/*/*.cc source/*/*/*.cc)
file(GLOB LIB_HEADERS include/*.h include/*/*.h include/*/*/*.h)
file(GLOB APP_SOURCES apps/*.cc apps/*/*.cc)
FILE(GLOB GTEST_FILES gtests/*cc)
FILE(GLOB BOOK_FILES fsi_book/*cc)

# Build both Release and Debug versions of library and executables
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")

      STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
  
      # Setup library
      SET(_lib "fsi-lib${_p}")
      MESSAGE("-- Configuring library ${_lib}")
      
      ADD_LIBRARY(${_lib} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
      if(CGAL_FOUND)
      target_link_libraries(${_lib} CGAL::CGAL)
      endif()

      if(ParMoonolith_FOUND)
      target_link_libraries(${_lib} ParMoonolith::par_moonolith)
      endif()

      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
      INSTALL(TARGETS ${_lib})
      SET(TEST_LIBRARIES_${_BUILD_TYPE} ${_lib})

      # Now setup all the executables
      foreach(sourcefile ${APP_SOURCES} )
          # I used a simple string replace, to cut off .cpp.
          get_filename_component(name ${sourcefile} NAME_WE)
          IF(${sourcefile} MATCHES "mpi")
              set(name mpi_${name})
          endif()
          # Add suffix
          set(name ${name}${_p})
          message("-- Configuring executable ${name}")
          add_executable( ${name} ${sourcefile} )
          # Make sure YourLib is linked to each app
          target_link_libraries(${name} ${_lib})
          DEAL_II_SETUP_TARGET(${name} ${_BUILD_TYPE})
          INSTALL(TARGETS ${name})
      endforeach( sourcefile ${APP_SOURCES} )

      # Google tests testing
      FIND_PACKAGE(GTest)
      IF(GTest_FOUND)
          include(GoogleTest)
          set(fsi_test "fsi_test${_p}")
          set(fsi_book "fsi_book${_p}")
          ADD_EXECUTABLE(${fsi_test} ${GTEST_FILES})
          TARGET_LINK_LIBRARIES(${fsi_test} ${GTEST_LIBRARIES} ${_lib})
          DEAL_II_SETUP_TARGET(${fsi_test} ${_BUILD_TYPE})
          INSTALL(TARGETS ${fsi_test})
          
          ADD_EXECUTABLE(${fsi_book} ${BOOK_FILES})
          TARGET_LINK_LIBRARIES(${fsi_book} ${GTEST_LIBRARIES} ${_lib})
          DEAL_II_SETUP_TARGET(${fsi_book} ${_BUILD_TYPE})
          INSTALL(TARGETS ${fsi_book})

          INCLUDE_DIRECTORIES(./include/ ${GTEST_INCLUDE_DIRS})

          ENABLE_TESTING()
          GTEST_DISCOVER_TESTS(${fsi_test})
          GTEST_DISCOVER_TESTS(${fsi_book})
          MESSAGE("-- Configured Gtest executable ${fsi_test}")
          MESSAGE("-- Configured Gtest executable ${fsi_book}")
      ENDIF(GTest_FOUND)
    endif()
endforeach()

# Deal.II style testing
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dealii_tests)
  ADD_SUBDIRECTORY(dealii_tests)
  ENABLE_TESTING()
  MESSAGE("-- Configured deal.II testing framework")
ENDIF()


# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
    MESSAGE("-- Configured Doxygen")
endif(DOXYGEN_FOUND)

add_custom_target(copyright
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/update_copyright.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Updating copyright information" VERBATIM)
MESSAGE("-- Configured Copyright")