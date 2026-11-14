#include "seam_carving.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace sc {

// ---------------------------------------------------------------------------
// Step (a): energy map
// ---------------------------------------------------------------------------
cv::Mat computeEnergy(const cv::Mat& image, EnergyMode mode) {
    CV_Assert(!image.empty());

    // Grayscale mode collapses the colours into one brightness channel first.
    // PerChannel mode leaves the image as it is, so Sobel runs on B, G and R separately.
    cv::Mat source = image;
    if (mode == EnergyMode::Grayscale && image.channels() == 3) {
        cv::cvtColor(image, source, cv::COLOR_BGR2GRAY);
    }

    // Sobel approximates the derivative in x and in y with a 3x3 kernel.
    // On a multi-channel image it filters every channel independently.
    // CV_64F keeps negative gradients, which an 8-bit output would clip to 0.
    cv::Mat gradX, gradY;
    cv::Sobel(source, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(source, gradY, CV_64F, 0, 1, 3);

    // e1 energy from the original paper, per channel: |dI/dx| + |dI/dy|.
    const cv::Mat perChannel = cv::abs(gradX) + cv::abs(gradY);

    // Add the channels together so the DP always receives one value per pixel.
    // (In Grayscale mode there is only one channel, so this is just a copy.)
    std::vector<cv::Mat> channels;
    cv::split(perChannel, channels);
    cv::Mat energy = channels[0].clone();
    for (size_t c = 1; c < channels.size(); ++c) {
        energy += channels[c];
    }
    return energy;
}

// ---------------------------------------------------------------------------
// Step (b): minimum seam by dynamic programming
// ---------------------------------------------------------------------------
Seam findVerticalSeamDP(const cv::Mat& energy) {
    CV_Assert(!energy.empty() && energy.type() == CV_64F);
    const int rows = energy.rows;
    const int cols = energy.cols;

    // M[i][j] = cost of the cheapest seam that starts anywhere in row 0 and ends at (i, j).
    // Base case: M[0][j] = e[0][j], which the clone already gives us.
    cv::Mat M = energy.clone();

    // Fill top to bottom: M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1]).
    for (int i = 1; i < rows; ++i) {
        const double* prev = M.ptr<double>(i - 1);
        double* cur = M.ptr<double>(i);
        for (int j = 0; j < cols; ++j) {
            double best = prev[j];
            if (j > 0)        best = std::min(best, prev[j - 1]);  // no left neighbour in column 0
            if (j + 1 < cols) best = std::min(best, prev[j + 1]);  // no right neighbour in last column
            cur[j] += best;  // cur[j] already holds e[i][j]
        }
    }

    // The cheapest seam ends at the smallest value in the bottom row.
    Seam seam(rows);
    const double* last = M.ptr<double>(rows - 1);
    int col = static_cast<int>(std::min_element(last, last + cols) - last);
    seam[rows - 1] = col;

    // Backtrack: the predecessor of (i, col) is whichever of the three cells above
    // has the smallest M, because that is the one the recurrence picked.
    // Ties prefer straight up, then left, then right.
    for (int i = rows - 1; i > 0; --i) {
        const double* prev = M.ptr<double>(i - 1);
        int bestCol = col;
        if (col > 0 && prev[col - 1] < prev[bestCol])        bestCol = col - 1;
        if (col + 1 < cols && prev[col + 1] < prev[bestCol]) bestCol = col + 1;
        col = bestCol;
        seam[i - 1] = col;
    }
    return seam;
}

// ---------------------------------------------------------------------------
// Alternative to step (b): greedy seam
// ---------------------------------------------------------------------------
Seam findVerticalSeamGreedy(const cv::Mat& energy) {
    CV_Assert(!energy.empty() && energy.type() == CV_64F);
    const int rows = energy.rows;
    const int cols = energy.cols;

    // Start at the cheapest pixel in the top row (the leftmost one if there is a tie).
    Seam seam(rows);
    const double* top = energy.ptr<double>(0);
    int col = static_cast<int>(std::min_element(top, top + cols) - top);
    seam[0] = col;

    // Walk down one row at a time. The only information used is the energy of the
    // three pixels directly below the current one: no table, and no looking ahead.
    // Ties prefer straight down, then left, then right (the same order as the DP backtrack).
    for (int i = 1; i < rows; ++i) {
        const double* row = energy.ptr<double>(i);
        int bestCol = col;
        if (col > 0 && row[col - 1] < row[bestCol])        bestCol = col - 1;
        if (col + 1 < cols && row[col + 1] < row[bestCol]) bestCol = col + 1;
        col = bestCol;
        seam[i] = col;
    }
    return seam;
}

double seamCost(const cv::Mat& energy, const Seam& seam) {
    CV_Assert(energy.type() == CV_64F && static_cast<int>(seam.size()) == energy.rows);
    double total = 0.0;
    for (int i = 0; i < energy.rows; ++i) {
        total += energy.at<double>(i, seam[i]);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Step (c): remove the seam
// ---------------------------------------------------------------------------
cv::Mat removeVerticalSeam(const cv::Mat& image, const Seam& seam) {
    CV_Assert(image.cols > 1 && static_cast<int>(seam.size()) == image.rows);

    cv::Mat out(image.rows, image.cols - 1, image.type());
    const size_t pixelBytes = image.elemSize();  // 3 for a BGR image, 1 for grayscale

    for (int i = 0; i < image.rows; ++i) {
        const int cut = seam[i];
        CV_Assert(cut >= 0 && cut < image.cols);
        const uchar* src = image.ptr(i);
        uchar* dst = out.ptr(i);
        // Copy the pixels left of the seam, then the pixels right of it shifted one place left.
        std::copy(src, src + cut * pixelBytes, dst);
        std::copy(src + (cut + 1) * pixelBytes, src + image.cols * pixelBytes,
                  dst + cut * pixelBytes);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Repeat until the image has the requested size
// ---------------------------------------------------------------------------
namespace {

cv::Mat reduceWidth(cv::Mat image, int targetWidth, const SeamFinder& findSeam,
                    EnergyMode energyMode, CarveStats* stats) {
    while (image.cols > targetWidth) {
        // The energy map is recomputed every time: removing a seam creates new
        // neighbours, so the old gradients next to the seam are no longer valid.
        const cv::Mat energy = computeEnergy(image, energyMode);
        const Seam seam = findSeam(energy);
        if (stats != nullptr) {
            stats->seamsRemoved += 1;
            stats->totalSeamEnergy += seamCost(energy, seam);
        }
        image = removeVerticalSeam(image, seam);
    }
    return image;
}

}  // namespace

cv::Mat seamCarve(const cv::Mat& image, int targetWidth, int targetHeight,
                  const SeamFinder& findSeam, EnergyMode energyMode, CarveStats* stats) {
    if (stats != nullptr) {
        *stats = CarveStats{};  // start from zero even if the caller reuses the struct
    }
    if (image.empty()) {
        throw std::invalid_argument("seamCarve: image is empty");
    }
    if (targetWidth < 1 || targetWidth > image.cols ||
        targetHeight < 1 || targetHeight > image.rows) {
        throw std::invalid_argument("seamCarve: target size must be between 1 and the image size");
    }

    cv::Mat result = reduceWidth(image.clone(), targetWidth, findSeam, energyMode, stats);

    // A horizontal seam of an image is a vertical seam of its transpose,
    // so the same code handles both directions.
    cv::Mat transposed = result.t();
    transposed = reduceWidth(transposed, targetHeight, findSeam, energyMode, stats);
    return transposed.t();
}

}  // namespace sc
