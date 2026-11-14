# Seam-Carving (Assignment 2)

Content-aware image resizing in C++17 with OpenCV (used only for image I/O, grayscale conversion and the Sobel filter).

## Layout

```
include/seam_carving.hpp      public functions
src/seam_carving.cpp          energy map, DP seam finder, seam removal, resize loop
src/main.cpp                  command-line app
tests/test_seam_carving.cpp   unit tests (run by CTest)
```

## Requirements

- CMake 3.16+ and a C++17 compiler
- OpenCV 4
  - Ubuntu: `sudo apt install libopencv-dev`
  - macOS: `brew install opencv`
  - Windows: `vcpkg install opencv4`, then pass vcpkg's toolchain file to CMake

## Build, test, run

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/seam_carve input.jpg output.jpg <target width> <target height> [rgb|gray]
```

The target size must be between 1 and the current size in each dimension (this version only shrinks images).

The output folder is created if it does not exist. The output file name needs an image extension such as `.png` or `.jpg`. An existing output file is overwritten, and the program prints a warning first.

## Energy function

The optional last argument selects how the energy map is computed:

| Option | Energy | Notes |
|---|---|---|
| `rgb` (default) | Sobel on each colour channel, summed over B, G and R | Detects edges between colours of equal brightness |
| `gray` | Sobel on the grayscale image | About a third of the filtering work, but blind to equal-brightness colour edges |

Example: `./build/seam_carve images/castle.png images/out/castle_gray.png 400 450 rgb`
