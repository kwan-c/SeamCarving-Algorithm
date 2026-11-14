# Seam-Carving (Assignment 2)

Content-aware image resizing in C++17 with OpenCV (used only for image I/O, grayscale conversion and the Sobel filter).

## Layout

```
include/seam_carving.hpp      public functions
src/seam_carving.cpp          energy map, DP and greedy seam finders, seam removal, resize loop
src/seam_overlay.cpp          paints the removed seams onto the original image
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
./build/seam_carve input.jpg output.jpg <target width> <target height> [dp|greedy] [rgb|gray] [seams=<image>]
```

The target size must be between 1 and the current size in each dimension (this version only shrinks images).

The output folder is created if it does not exist. The output file name needs an image extension such as `.png` or `.jpg`. An existing output file is overwritten, and the program prints a warning first.

## Options

Option keywords go after the four required arguments, in any order.

### Seam-finding algorithm

| Option | Algorithm | Notes |
|---|---|---|
| `dp` (default) | Dynamic programming | Always finds the minimum-energy seam |
| `greedy` | Row-by-row greedy choice | Finds a seam faster, but not always the cheapest one |

After each run the program prints the number of seams removed and their total energy, so the algorithms can be compared.

### Seam overlay

Add `seams=<image>` to also save the original image with every removed seam painted on it: vertical seams in red, horizontal seams in yellow. Horizontal seams are found after the width has been reduced, so they show gaps where a vertical seam had already removed the pixel.

Example: `./build/seam_carve images/castle.png images/out/castle_dp.png 186 140 dp seams=images/out/castle_dp_seams.png`

Painting the seams adds a little bookkeeping to the run, so leave `seams=` off when measuring running time.

### Energy function

| Option | Energy | Notes |
|---|---|---|
| `rgb` (default) | Sobel on each colour channel, summed over B, G and R | Detects edges between colours of equal brightness |
| `gray` | Sobel on the grayscale image | About a third of the filtering work, but blind to equal-brightness colour edges |

Example: `./build/seam_carve images/castle.png images/out/castle_gray.png 400 450 dp rgb`
