# Seam-Carving (Assignment 2)

[![CI](https://github.com/kwan-c/AlgorithmA2_SeamCarving/actions/workflows/ci.yml/badge.svg)](https://github.com/kwan-c/AlgorithmA2_SeamCarving/actions/workflows/ci.yml)

Content-aware image resizing in C++17 with OpenCV (used only for image I/O, grayscale conversion and the Sobel filter).

## Layout

```
include/seam_carving.hpp      public functions
src/seam_carving.cpp          energy map, DP and greedy seam finders, seam removal, resize loop
src/seam_graph_cut.cpp        graph cut seam finder (builds the flow network, reads the seam off the cut)
src/max_flow.cpp              flow network with two maximum-flow algorithms (Edmonds-Karp and Dinic)
src/seam_overlay.cpp          visualization: paints the removed seams onto the original image, draws step-by-step frames
src/main.cpp                  command-line app
tests/test_seam_carving.cpp   unit tests (run by CTest)
.github/workflows/ci.yml      continuous integration: build, test and sample run on every push
```

## Requirements

- CMake 3.16+ and a C++17 compiler
- OpenCV 4
  - Ubuntu: `sudo apt install libopencv-dev`
  - macOS: `brew install opencv`
  - Windows: `vcpkg install opencv4`, then pass vcpkg's toolchain file to CMake
- Optional: ffmpeg, only for joining step-by-step frames into a video (see [Step-by-step frames](#step-by-step-frames)). The program itself does not need it.
  - Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Build, test, run

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/seam_carve <image name> <target width> <target height> [dp|greedy|graphcut|graphcut-dinic] [rgb|gray] [seams] [frames]
```

The target size must be between 1 and the current size in each dimension (this version only shrinks images).

### Where images go

- **Input:** put the image in the `images` folder and give its file name, for example `castle.png`. `images/castle.png` is accepted too. Images in any other folder are refused.
- **Output:** results are always written to `images/out`, which is created if it does not exist. The file name is built from the image name and the settings, for example `castle_186x140_dp_rgb.png`, so runs with different settings never overwrite each other. Outputs are always PNG.
- **Run from the project folder.** The `images` folder is looked up relative to the folder the program is run in.
- Running the same settings again overwrites the earlier result, and the program prints a warning first.

Example: `./build/seam_carve castle.png 400 450` writes `images/out/castle_400x450_dp_rgb.png`.

## Options

Option keywords go after the three required arguments, in any order.

### Seam-finding algorithm

| Option | Algorithm | Notes |
|---|---|---|
| `dp` (default) | Dynamic programming | Always finds the minimum-energy seam |
| `greedy` | Row-by-row greedy choice | Finds a seam faster, but not always the cheapest one |
| `graphcut` | Minimum cut by maximum flow (Edmonds-Karp) | Finds a minimum-energy seam like `dp`, but is far slower; use a small image or a small size change |
| `graphcut-dinic` | The same minimum cut, with Dinic's max-flow algorithm | Finds exactly the same seams as `graphcut`, several times faster; still far slower than `dp` |

After each run the program prints the number of seams removed and their total energy, so the algorithms can be compared.

### Seam overlay

Add `seams` to also save the original image with every removed seam painted on it: vertical seams in red, horizontal seams in yellow. It is saved next to the result with `_seams` added to the name. Horizontal seams are found after the width has been reduced, so they show gaps where a vertical seam had already removed the pixel.

Example: `./build/seam_carve castle.png 400 450 dp seams` writes `images/out/castle_400x450_dp_rgb.png` and `images/out/castle_400x450_dp_rgb_seams.png`.

Painting the seams adds a little bookkeeping to the run, so leave `seams` off when measuring running time.

### Step-by-step frames

Add `frames` to also save one picture per seam, showing each seam just before it is removed (vertical seams in red, horizontal seams in yellow), followed by one picture of the final image. The pictures go in a folder next to the result, named after it with `_frames` added, as `frame_0001.png`, `frame_0002.png` and so on. Every frame has the size of the original image, with the shrinking picture in its top-left corner.

Example: `./build/seam_carve castle.png 400 450 dp frames` writes the frames to `images/out/castle_400x450_dp_rgb_frames/`.

To join the frames into a video, install ffmpeg (see [Requirements](#requirements)) and run:

```
ffmpeg -framerate 30 -i images/out/castle_400x450_dp_rgb_frames/frame_%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p images/out/castle_steps.mp4
```

Writing the frames makes the run several times slower, so leave `frames` off when measuring running time.

### Energy function

| Option | Energy | Notes |
|---|---|---|
| `rgb` (default) | Sobel on each colour channel, summed over B, G and R | Detects edges between colours of equal brightness |
| `gray` | Sobel on the grayscale image | About a third of the filtering work, but blind to equal-brightness colour edges |

Example: `./build/seam_carve castle.png 400 450 dp gray` writes `images/out/castle_400x450_dp_gray.png`.

## Continuous integration

Every push to `main` and every pull request runs `.github/workflows/ci.yml` on GitHub Actions. On a clean Ubuntu machine it installs OpenCV, builds the project, runs all the unit tests, and then runs the program on `images/castle.png` with the DP, greedy and graph cut algorithms. A failing test marks the commit as failed. The resized images and seam overlays from that run can be downloaded from the run's page as the `sample-results` artifact.
