#include "seam_overlay.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sc {

cv::Mat seamCarveWithOverlay(const cv::Mat& image, int targetWidth, int targetHeight,
                             cv::Mat& overlay, const SeamFinder& findSeam,
                             EnergyMode energyMode, CarveStats* stats,
                             const SeamObserver& alsoNotify) {
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
    const SeamObserver paintSeam = [&](const cv::Mat& current, const Seam& seam,
                                       bool horizontal) {
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

        if (alsoNotify) {
            alsoNotify(current, seam, horizontal);  // pass the seam on to the caller's observer
        }
    };

    return seamCarve(image, targetWidth, targetHeight, findSeam, energyMode, stats, paintSeam);
}

cv::Mat drawSeamFrame(const cv::Mat& image, const Seam& seam, bool horizontal,
                      cv::Size canvasSize) {
    if (image.empty()) {
        throw std::invalid_argument("drawSeamFrame: image is empty");
    }
    if (image.type() != CV_8UC3 && image.type() != CV_8UC1) {
        throw std::invalid_argument("drawSeamFrame: image must be 8-bit, 1 or 3 channels");
    }
    if (!seam.empty() && static_cast<int>(seam.size()) != image.rows) {
        throw std::invalid_argument("drawSeamFrame: the seam needs one entry per image row");
    }

    // Work on a colour copy so the caller's image is untouched and the paint is visible.
    cv::Mat picture;
    if (image.channels() == 1) {
        cv::cvtColor(image, picture, cv::COLOR_GRAY2BGR);
    } else {
        picture = image.clone();
    }

    // Paint the seam while it is still vertical: one pixel in every row.
    const cv::Vec3b colour = horizontal ? kHorizontalSeamColour : kVerticalSeamColour;
    for (int i = 0; i < static_cast<int>(seam.size()); ++i) {
        if (seam[i] < 0 || seam[i] >= picture.cols) {
            throw std::invalid_argument("drawSeamFrame: seam column outside the image");
        }
        picture.at<cv::Vec3b>(i, seam[i]) = colour;
    }

    // During the height pass the image is transposed; turn it back so it looks upright.
    // The painted seam turns with it and becomes a horizontal line.
    if (horizontal) {
        cv::Mat upright;
        cv::transpose(picture, upright);
        picture = upright;
    }

    if (picture.cols > canvasSize.width || picture.rows > canvasSize.height) {
        throw std::invalid_argument("drawSeamFrame: the image does not fit in the canvas");
    }

    // Same-sized canvas for every frame; the shrinking picture sits in its top-left corner.
    cv::Mat frame(canvasSize, CV_8UC3,
                  cv::Scalar(kFrameBackgroundColour[0], kFrameBackgroundColour[1],
                             kFrameBackgroundColour[2]));
    picture.copyTo(frame(cv::Rect(0, 0, picture.cols, picture.rows)));
    return frame;
}

}  // namespace sc
