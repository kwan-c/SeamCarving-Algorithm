#pragma once

#include <functional>
#include <vector>

#include <opencv2/core.hpp>

namespace sc {

// A vertical seam: seam[i] is the column of the pixel removed from row i.
// A valid seam has one entry per row, and neighbouring entries differ by at most 1.
using Seam = std::vector<int>;

// Any function that takes an energy map and returns a vertical seam.
// The DP finder is the default; the greedy and graph-cut finders will plug in here later.
using SeamFinder = std::function<Seam(const cv::Mat& energy)>;

// How the energy map treats colour.
enum class EnergyMode {
    PerChannel,  // default: Sobel on each colour channel, then sum the channels
    Grayscale    // alternative: convert to grayscale first, then one Sobel pass
};

// Energy map e[i][j] = |dI/dx| + |dI/dy| using 3x3 Sobel filters.
// PerChannel adds this up over the B, G and R channels, so it also sees edges between
// two colours of equal brightness, which Grayscale misses. Grayscale is about 3x cheaper.
// Returns a single-channel CV_64F matrix with the same size as the image.
cv::Mat computeEnergy(const cv::Mat& image, EnergyMode mode = EnergyMode::PerChannel);

// Minimum-energy vertical seam by dynamic programming.
// `energy` must be a non-empty CV_64F matrix. Time O(rows * cols), space O(rows * cols).
Seam findVerticalSeamDP(const cv::Mat& energy);

// Sum of the energy values along a seam (used for testing and for comparing algorithms).
double seamCost(const cv::Mat& energy, const Seam& seam);

// Returns a copy of `image` that is one column narrower, with the seam pixels deleted.
cv::Mat removeVerticalSeam(const cv::Mat& image, const Seam& seam);

// Shrinks the image to targetWidth x targetHeight by repeatedly removing seams.
// Vertical seams are removed first, then horizontal seams (via the transposed image).
// Throws std::invalid_argument if the target is larger than the image or smaller than 1.
cv::Mat seamCarve(const cv::Mat& image, int targetWidth, int targetHeight,
                  const SeamFinder& findSeam = findVerticalSeamDP,
                  EnergyMode energyMode = EnergyMode::PerChannel);

}  // namespace sc
