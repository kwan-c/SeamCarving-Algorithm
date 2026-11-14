#pragma once

#include <opencv2/core.hpp>

#include "seam_carving.hpp"

namespace sc {

// Colours used to paint removed pixels, in OpenCV's blue-green-red order.
const cv::Vec3b kVerticalSeamColour(0, 0, 255);      // red
const cv::Vec3b kHorizontalSeamColour(0, 255, 255);  // yellow
const cv::Vec3b kFrameBackgroundColour(40, 40, 40);  // dark gray, behind a step-by-step frame

// Resizes exactly like seamCarve and also fills in `overlay`: a copy of the ORIGINAL
// image, at its original size, on which every pixel that the resize removed is painted
// (vertical seams red, horizontal seams yellow).
//
// Horizontal seams are found after the width has been reduced, so on the overlay they
// show gaps in the columns that a vertical seam had already taken.
//
// The image must be 8-bit with 1 or 3 channels. The overlay is always 3-channel.
// If `alsoNotify` is set, every seam is passed on to it as well (after it has been
// painted), so a caller can watch the same resize, for example to save frames.
cv::Mat seamCarveWithOverlay(const cv::Mat& image, int targetWidth, int targetHeight,
                             cv::Mat& overlay,
                             const SeamFinder& findSeam = findVerticalSeamDP,
                             EnergyMode energyMode = EnergyMode::PerChannel,
                             CarveStats* stats = nullptr,
                             const SeamObserver& alsoNotify = nullptr);

// Draws one step of a resize, for a step-by-step visualization: the image as it is at
// that moment, with the seam that is about to be removed painted on it (vertical seams
// red, horizontal seams yellow).
//
// The arguments are exactly what a SeamObserver receives. During the height pass the
// image arrives transposed; the frame is turned back so it always looks upright.
// The picture is placed in the top-left corner of a canvas of `canvasSize`, so that
// every frame of one resize has the same size and the frames can be joined into a video.
// Pass an empty seam to draw the image with nothing painted (used for the final frame).
//
// The image must be 8-bit with 1 or 3 channels and fit inside the canvas.
cv::Mat drawSeamFrame(const cv::Mat& image, const Seam& seam, bool horizontal,
                      cv::Size canvasSize);

}  // namespace sc
