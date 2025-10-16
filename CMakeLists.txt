cmake_minimum_required(VERSION 3.10)

# Define el nombre del proyecto y el lenguaje principal.
project(proyecto_computacion_I CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- RUTAS Y ARCHIVOS FUENTE ---

set(CPP_ROOT_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

include_directories(
        ${CPP_ROOT_DIR}/include
)

set(PROJECT_SOURCES
        ${CPP_ROOT_DIR}/src/jni/jni_interface.cpp
        ${CPP_ROOT_DIR}/src/physics/manning_solver.cpp
        ${CPP_ROOT_DIR}/test/physics/manning_solver_test.cpp
)
