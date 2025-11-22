# Usamos una versión más moderna para asegurar un buen soporte de JNI y CUDA
cmake_minimum_required(VERSION 3.18)

# Le decimos a CMake que "finja" compilar para la arquitectura de GPU nativa (local).
set(CMAKE_CUDA_ARCHITECTURES "75")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Le decimos a CLion que este proyecto usa C++ y CUDA
project(projectstalker_ide CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- RUTAS Y PAQUETES ---

# 1. Busca JNI de forma robusta (no depende de $ENV{JAVA_HOME})
find_package(JNI REQUIRED)

# 2. Añadimos solo las cabeceras de nuestro proyecto
set(CPP_ROOT_DIR ${CMAKE_SOURCE_DIR}/native-lib/src/main/cpp)
include_directories(
        ${CPP_ROOT_DIR}/include
        "/usr/local/cuda/include"
)


# --- DEFINICIÓN DEL TARGET (LA "LIBRERÍA FALSA") ---

# Agrupamos todos los archivos fuente
file(GLOB_RECURSE PROJECT_SOURCES
        "${CPP_ROOT_DIR}/src/*.cpp"
        "${CPP_ROOT_DIR}/src/*.cu"
        "${CPP_ROOT_DIR}/test/*.cpp"
)

# Creamos la librería falsa
add_library(native_lib_ide SHARED ${PROJECT_SOURCES}
        native-lib/src/main/cpp/include/projectstalker/physics/manning_kernel.h
        native-lib/src/main/cpp/include/projectstalker/physics/transport_kernel.h
        native-lib/src/main/cpp/src/physics/transport_kernel.cu)

# 3. CONECTAMOS JNI
# Le decimos a nuestro target falso que use las cabeceras de JNI
target_link_libraries(native_lib_ide PRIVATE JNI::JNI)