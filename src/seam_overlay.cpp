#include "seam_overlay.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sc {

cv::Mat seamCarveWithOverlay(const cv::Mat& image, int targetWidth, int targetHeight,
                             cv::Mat& overlay, const SeamFinder& findSeam,
                             EnergyMode energyMode, CarveStats* stats) {
    if (image.empty()) {
        throw std::invalid_argument("seamCarveWithOverlay: image is empty");
    }
    if (image.type() != CV_8UC3 && image.type() != CV_8UC1) {
        throw std::invalid_argument("seamCarveWithOverlay: image must be 8-bit, 1 or 3 channels");
    }

    // The overlay starts as the original picture, in colour so the paint is visible.
    if (image.channels() == 1) {
        cv::cvtColor(image, overlay, cv::COLOR_GRAY2BGR);
    } else {
        overlay = image.clone();
    }

    // The difficulty: a seam is found in the current, already-shrunk image, so its column
    // numbers no longer match the original picture. To translate back, we carry a second
    // "image" through the resize whose pixels are coordinates instead of colours:
    //   origin(i, j) = (row, col) that the pixel now at (i, j) had in the original.
    // Whatever seam is removed from the picture is removed from `origin` as well, so the
    // two always stay the same shape and each pixel keeps its original address.
    cv::Mat origin(image.rows, image.cols, CV_32SC2);
    for (int i = 0; i < origin.rows; ++i) {
        for (int j = 0; j < origin.cols; ++j) {
            origin.at<cv::Vec2i>(i, j) = cv::Vec2i(i, j);
        }
    }
    bool originIsTransposed = false;

    // Called by seamCarve once per seam, just before the seam is removed.
    const SeamObserver paintSeam = [&](const cv::Mat&, const Seam& seam, bool horizontal) {
        // seamCarve transposes the picture before the height pass, so do the same to
        // `origin` the first time a horizontal seam arrives.
        if (horizontal && !originIsTransposed) {
            cv::Mat flipped;
            cv::transpose(origin, flipped);
            origin = flipped;
            originIsTransposed = true;
        }

        const cv::Vec3b colour = horizontal ? kHorizontalSeamColour : kVerticalSeamColour;
        for (int i = 0; i < origin.rows; ++i) {
            const cv::Vec2i where = origin.at<cv::Vec2i>(i, seam[i]);  // (row, col) in the original
            overlay.at<cv::Vec3b>(where[0], where[1]) = colour;
        }

        // Keep `origin` in step with the picture. removeVerticalSeam copies whole pixels
        // by their size in bytes, so it works on coordinate pairs as well as on colours.
        origin = removeVerticalSeam(origin, seam);
    };

    return seamCarve(image, targetWidth, targetHeight, findSeam, energyMode, stats, paintSeam);
}

}  // namespace sc
