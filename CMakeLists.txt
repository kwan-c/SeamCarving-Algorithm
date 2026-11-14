cmake_minimum_required(VERSION 3.16)
project(SeamCarving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimised build so timing measurements are meaningful.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# The algorithms live in a library so the app and the tests share the same code.
add_library(seamcarving src/seam_carving.cpp)
target_include_directories(seamcarving PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(seamcarving PUBLIC ${OpenCV_LIBS})
if(NOT MSVC)
  target_compile_options(seamcarving PRIVATE -Wall -Wextra)
endif()

add_executable(seam_carve src/main.cpp)
target_link_libraries(seam_carve PRIVATE seamcarving)

enable_testing()
add_executable(test_seam_carving tests/test_seam_carving.cpp)
target_link_libraries(test_seam_carving PRIVATE seamcarving)
add_test(NAME seam_carving_tests COMMAND test_seam_carving)
