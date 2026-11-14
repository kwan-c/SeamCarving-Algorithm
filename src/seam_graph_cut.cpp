#include "seam_carving.hpp"

#include <stdexcept>
#include <vector>

#include "max_flow.hpp"

namespace sc {

namespace {

// Shared by all the public functions. `lastColumnRemovable` selects the sink edges:
//   true   last-column pixel -> sink has capacity e[i][last], so that pixel can be the seam
//   false  last-column pixel -> sink is infinite, as the assignment brief words it
// `algorithm` selects how the maximum flow is computed; the resulting seam is the same.
Seam graphCutSeam(const cv::Mat& energy, bool lastColumnRemovable, MaxFlowAlgorithm algorithm) {
    CV_Assert(!energy.empty() && energy.type() == CV_64F);
    const int rows = energy.rows;
    const int cols = energy.cols;

    if (cols == 1) {
        if (!lastColumnRemovable) {
            throw std::invalid_argument(
                "findVerticalSeamGraphCutAsDescribed: needs at least two columns");
        }
        return Seam(rows, 0);  // only one possible seam, no network needed
    }

    // One node per pixel, then the source and the sink.
    const int source = rows * cols;
    const int sink = rows * cols + 1;
    const auto node = [cols](int i, int j) { return i * cols + j; };

    // "Infinite" capacity: any number larger than every possible finite cut will do.
    // The sum of all energies plus one is safely larger, and unlike a true infinity it
    // can be subtracted from without producing NaN.
    const double infinite = cv::sum(energy)[0] + 1.0;

    FlowNetwork network(rows * cols + 2);
    for (int i = 0; i < rows; ++i) {
        const double* e = energy.ptr<double>(i);

        // The left edge of the image belongs to the source side.
        network.addEdge(source, node(i, 0), infinite);

        for (int j = 0; j < cols; ++j) {
            if (j + 1 < cols) {
                // Forward edge: cutting it removes pixel (i, j), at the price of its energy.
                network.addEdge(node(i, j), node(i, j + 1), e[j]);
                // Backward edge: if (i, j+1) is on the source side then so is (i, j).
                // This makes the cut cross each row exactly once.
                network.addEdge(node(i, j + 1), node(i, j), infinite);
            } else {
                // Last column: the edge into the sink plays the role of the forward edge.
                network.addEdge(node(i, j), sink, lastColumnRemovable ? e[j] : infinite);
            }

            // Diagonal edges: if (i, j) is on the source side then so are (i-1, j-1) and
            // (i+1, j-1). This keeps the cut in neighbouring rows at most one column apart.
            if (j > 0) {
                if (i > 0)        network.addEdge(node(i, j), node(i - 1, j - 1), infinite);
                if (i + 1 < rows) network.addEdge(node(i, j), node(i + 1, j - 1), infinite);
            }
        }
    }

    network.maxFlow(source, sink, algorithm);
    const std::vector<bool> onSourceSide = network.sourceSide(source);

    // The seam pixel in each row is the rightmost pixel on the source side: the forward
    // edge leaving it is the one the minimum cut crosses in that row.
    Seam seam(rows);
    for (int i = 0; i < rows; ++i) {
        int j = cols - 1;
        while (j > 0 && !onSourceSide[node(i, j)]) {
            --j;
        }
        seam[i] = j;  // column 0 is always on the source side
    }
    return seam;
}

}  // namespace

Seam findVerticalSeamGraphCut(const cv::Mat& energy) {
    return graphCutSeam(energy, true, MaxFlowAlgorithm::EdmondsKarp);
}

Seam findVerticalSeamGraphCutDinic(const cv::Mat& energy) {
    return graphCutSeam(energy, true, MaxFlowAlgorithm::Dinic);
}

Seam findVerticalSeamGraphCutAsDescribed(const cv::Mat& energy) {
    return graphCutSeam(energy, false, MaxFlowAlgorithm::EdmondsKarp);
}

}  // namespace sc
