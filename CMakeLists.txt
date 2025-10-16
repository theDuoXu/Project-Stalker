# Este CMakeLists.txt es SOLO para que CLion entienda la estructura del proyecto
# y proporcione análisis de código. LA COMPILACIÓN REAL LA HACE GRADLE.
cmake_minimum_required(VERSION 3.10)

# Le decimos a CLion el nombre del proyecto. Los lenguajes son para el análisis.
project(projectstalker_ide CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- RUTAS ---
# Le damos a CLion las rutas a los headers para que encuentre las definiciones
set(CPP_ROOT_DIR ${CMAKE_SOURCE_DIR}/native-lib/src/main/cpp)
include_directories(
        ${CPP_ROOT_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/linux
)

# --- DEFINICIÓN DEL TARGET (LA "LIBRERÍA FALSA") ---

# Agrupamos TODOS los archivos de código fuente (.cpp y .cu)
# para que CLion los reconozca como parte del mismo "target".
file(GLOB_RECURSE PROJECT_SOURCES
        "${CPP_ROOT_DIR}/src/*.cpp"
        "${CPP_ROOT_DIR}/src/*.cu"
        "${CPP_ROOT_DIR}/test/*.cpp"
)

# Creamos una librería falsa llamada 'native_lib_ide'
# CLion ahora asociará todos los archivos de PROJECT_SOURCES a este target.
add_library(native_lib_ide SHARED ${PROJECT_SOURCES})