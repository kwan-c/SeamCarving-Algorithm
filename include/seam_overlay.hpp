#pragma once

#include <opencv2/core.hpp>

#include "seam_carving.hpp"

namespace sc {

// Colours used to paint removed pixels, in OpenCV's blue-green-red order.
const cv::Vec3b kVerticalSeamColour(0, 0, 255);      // red
const cv::Vec3b kHorizontalSeamColour(0, 255, 255);  // yellow

// Resizes exactly like seamCarve and also fills in `overlay`: a copy of the ORIGINAL
// image, at its original size, on which every pixel that the resize removed is painted
// (vertical seams red, horizontal seams yellow).
//
// Horizontal seams are found after the width has been reduced, so on the overlay they
// show gaps in the columns that a vertical seam had already taken.
//
// The image must be 8-bit with 1 or 3 channels. The overlay is always 3-channel.
cv::Mat seamCarveWithOverlay(const cv::Mat& image, int targetWidth, int targetHeight,
                             cv::Mat& overlay,
                             const SeamFinder& findSeam = findVerticalSeamDP,
                             EnergyMode energyMode = EnergyMode::PerChannel,
                             CarveStats* stats = nullptr);

}  // namespace sc
