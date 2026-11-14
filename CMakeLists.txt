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
add_library(seamcarving
  src/seam_carving.cpp
  src/seam_graph_cut.cpp
  src/max_flow.cpp
  src/seam_overlay.cpp
)
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

# Each name below is one test in tests/test_seam_carving.cpp. Registering them one by one
# makes CTest report every test on its own line, so a failure points at a specific test.
set(SEAM_CARVING_TESTS
  dp_hand_worked_example
  dp_matches_brute_force
  dp_edge_cases
  greedy_hand_worked_example
  greedy_matches_dp_on_easy_input
  greedy_never_beats_dp
  greedy_edge_cases
  maxflow_textbook_network
  maxflow_simple_cases
  maxflow_dinic_matches_edmonds_karp
  graphcut_hand_worked_example
  graphcut_matches_dp_cost
  graphcut_edge_cases
  graphcut_as_described_skips_last_column
  graphcut_dinic_same_seam
  remove_vertical_seam
  energy_both_modes
  energy_equal_brightness_colour_edge
  resize_reaches_target_size
  resize_statistics
  resize_rejects_invalid_input
  resize_observer_sees_every_seam
  overlay_vertical_seams
  overlay_both_directions
  overlay_matches_plain_resize
  overlay_input_checks
  overlay_passes_seams_on
  frame_shows_vertical_seam
  frame_shows_horizontal_seam
  frame_input_checks
)
foreach(test_name IN LISTS SEAM_CARVING_TESTS)
  add_test(NAME ${test_name} COMMAND test_seam_carving ${test_name})
endforeach()

# Fails if the list above and the list inside the test program ever get out of step.
add_test(NAME test_list_is_complete
         COMMAND test_seam_carving --check-names ${SEAM_CARVING_TESTS})
