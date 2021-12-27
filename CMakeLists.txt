CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(gtest)

FIND_PACKAGE(deal.II 9.3 REQUIRED
HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Fix warning on mac
SET(CMAKE_MACOSX_RPATH 1)

# Common library files
file(GLOB LIB_SOURCES source/*.cc )
file(GLOB LIB_HEADERS lib/*.h )
ADD_LIBRARY(fsi-lib SHARED ${LIB_SOURCES} ${LIB_HEADERS})
DEAL_II_SETUP_TARGET(fsi-lib)

# Variuous executables
file(GLOB APP_SOURCES apps/*.cc )
foreach(sourcefile ${APP_SOURCES} )
    # I used a simple string replace, to cut off .cpp.
    get_filename_component(name ${sourcefile} NAME_WE)
    message("Preparing executable ${name}")
    add_executable( ${name} ${sourcefile} )
    # Make sure YourLib is linked to each app
    target_link_libraries( ${name} fsi-lib )
    DEAL_II_SETUP_TARGET(${name})
endforeach( sourcefile ${APP_SOURCES} )

FIND_PACKAGE(GTest)
FILE(GLOB test_files tests/*cc)
ADD_EXECUTABLE(gtest ${test_files})
TARGET_LINK_LIBRARIES(gtest ${GTEST_LIBRARY} fsi-lib)   
DEAL_II_SETUP_TARGET(gtest)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS} ./include/)

ENABLE_TESTING()
GTEST_DISCOVER_TESTS(gtest)

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
    MESSAGE("-- Configured Doxygen")
endif(DOXYGEN_FOUND)