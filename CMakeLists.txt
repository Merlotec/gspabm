cmake_minimum_required(VERSION 3.18)
project(gspabm LANGUAGES CXX CUDA)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the CUDA architecture(s) you want to support
# Adjust the architectures according to your target GPUs
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add the executable or library
add_executable(gspabmexec
        src/main.cu
        src/peris.cu
        src/world.cu)

add_definitions(-DPERIS_DISPLAY)

# Set properties for the target
set_target_properties(gspabmexec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gspabmexec PRIVATE include/)

target_link_libraries(gspabmexec sfml-graphics sfml-window sfml-system)