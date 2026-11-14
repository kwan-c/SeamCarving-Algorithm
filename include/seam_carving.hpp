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

// Greedy alternative to the DP. Starts at the smallest pixel in the top row, then for
// every following row steps to the smallest of the (up to) three pixels below it.
// It never looks further than one row ahead, so the seam is always valid but is not
// guaranteed to be the cheapest. Time O(rows + cols), extra space O(rows) for the seam.
Seam findVerticalSeamGreedy(const cv::Mat& energy);

// Graph cut alternative to the DP. Builds a flow network with one node per pixel, runs
// maximum flow (Edmonds-Karp), and reads the seam off the minimum cut: in each row the
// seam pixel is the rightmost pixel on the source side. The result is a minimum-energy
// seam, so its cost always equals the DP's; when several seams tie, the two algorithms
// may pick different ones. Time O(V * E^2) with V = rows * cols + 2 and E about 4 * V,
// which is far slower than the DP. (Implemented in seam_graph_cut.cpp.)
Seam findVerticalSeamGraphCut(const cv::Mat& energy);

// The same network and the same seam as findVerticalSeamGraphCut, but the maximum flow
// is computed with Dinic's algorithm instead of Edmonds-Karp. Worst case O(V^2 * E),
// and in practice much faster on these grid-shaped networks.
Seam findVerticalSeamGraphCutDinic(const cv::Mat& energy);

// The network exactly as the assignment brief words it, where every last-column pixel
// is tied to the sink with infinite capacity. A pixel in the last column can then never
// be the seam pixel, so this version finds the cheapest seam that avoids the last
// column, which is not always the true minimum. Kept to demonstrate that difference;
// needs at least two columns.
Seam findVerticalSeamGraphCutAsDescribed(const cv::Mat& energy);

// Sum of the energy values along a seam (used for testing and for comparing algorithms).
double seamCost(const cv::Mat& energy, const Seam& seam);

// Returns a copy of `image` that is one column narrower, with the seam pixels deleted.
cv::Mat removeVerticalSeam(const cv::Mat& image, const Seam& seam);

// Numbers collected during a resize, for comparing seam-finding algorithms.
struct CarveStats {
    int seamsRemoved = 0;          // vertical plus horizontal seams
    double totalSeamEnergy = 0.0;  // sum of seamCost over every removed seam
};

// An optional function that seamCarve calls once for every seam, just before removing it.
// It lets other code watch the resize (for example to draw the seams) without the
// resize loop knowing anything about drawing.
//   image       the current image, oriented so that the seam is vertical; while
//               horizontal seams are being removed this is the transposed image
//   seam        the seam that is about to be removed from `image`
//   horizontal  false during the width pass, true during the height pass
using SeamObserver =
    std::function<void(const cv::Mat& image, const Seam& seam, bool horizontal)>;

// Shrinks the image to targetWidth x targetHeight by repeatedly removing seams.
// Vertical seams are removed first, then horizontal seams (via the transposed image).
// Throws std::invalid_argument if the target is larger than the image or smaller than 1.
// If `stats` is not null, it is reset and filled in with the totals for this resize.
// If `observer` is set, it is called once per seam as described above.
cv::Mat seamCarve(const cv::Mat& image, int targetWidth, int targetHeight,
                  const SeamFinder& findSeam = findVerticalSeamDP,
                  EnergyMode energyMode = EnergyMode::PerChannel,
                  CarveStats* stats = nullptr,
                  const SeamObserver& observer = nullptr);

}  // namespace sc
