cmake_minimum_required(VERSION 4.0)
project(proyecto_computacion_I)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/main/cpp/include)
include_directories(src/main/cpp/include/projectstalker)
include_directories(src/main/cpp/include/projectstalker/physics)

add_executable(proyecto_computacion_I
        src/main/cpp/include/projectstalker/physics/manning_kernel.h
        src/main/cpp/include/projectstalker/physics/manning_solver.h
        src/main/cpp/src/jni/jni_interface.cpp
        src/main/cpp/src/physics/manning_kernel.cu
        src/main/cpp/src/physics/manning_solver.cpp
        src/main/cpp/test/physics/manning_solver_test.cpp)
