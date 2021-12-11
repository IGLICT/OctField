cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

# Project
set(PROJECT_NAME IMP_SAMPLING)
set(CMAKE_CXX_STANDARD 11)
get_filename_component(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
set(MANIFOLD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/manifold)
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)

# libigl
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(LIBIGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/libigl)
option(LIBIGL_WITH_OPENGL                   "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW              "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI        "Use GLFW IMGUI"     ON)

find_package(LIBIGL REQUIRED QUIET)


# includes
include_directories(${EXTERNAL_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${LIBIGL_DIR}/include)
include_directories(${LIBIGL_DIR}/external/glad/include)
include_directories(${LIBIGL_DIR}/external)
include_directories(${LIBIGL_DIR}/external/libigl-imgui)
include_directories(${LIBIGL_DIR}/external/imgui/examples)
include_directories(${EIGEN_DIR})
include_directories(${MANIFOLD_DIR})


# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)


# Source Files
file(GLOB SRC_FILES ${PROJECT_DIR}/src/*.cpp)
file(GLOB HEADER_FILES ${PROJECT_DIR}/src/*.h)
file(GLOB MANIFOLD_FILES ${MANIFOLD_DIR}/*.cpp)

add_subdirectory(${PYBIND11_DIR} pybind11)
pybind11_add_module(imp_samplingv1 ${PROJECT_DIR}/src/imp_sampling.cpp ${MANIFOLD_FILES})
target_link_libraries(imp_samplingv1 PRIVATE igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui)



