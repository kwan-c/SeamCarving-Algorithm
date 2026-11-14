// Usage: seam_carve <input image> <output image> <target width> <target height> [rgb|gray]
//
// The optional last argument chooses the energy function:
//   rgb   Sobel on each colour channel, summed (default)
//   gray  Sobel on the grayscale image

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include <opencv2/imgcodecs.hpp>

#include "seam_carving.hpp"

static void printUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " <input> <output> <target width> <target height> [rgb|gray]\n"
              << "  rgb   energy from each colour channel, summed (default)\n"
              << "  gray  energy from the grayscale image\n"
              << "The output folder is created if it does not exist.\n"
              << "An existing output file is overwritten, with a warning.\n";
}

// Prints a warning if writing to `outputPath` would replace an existing file.
// It only warns; the run continues, and the file is replaced once the run succeeds.
static void warnIfOverwriting(const std::string& inputPath, const std::string& outputPath) {
    // The error_code overloads return false instead of throwing if a path cannot be checked.
    std::error_code error;
    if (!std::filesystem::is_regular_file(outputPath, error)) {
        return;  // nothing there yet (or not a file), so nothing will be overwritten
    }

    // equivalent() is true when both paths lead to the same file on disk,
    // even if they are spelled differently (for example "a.png" and "./a.png").
    if (std::filesystem::equivalent(inputPath, outputPath, error)) {
        std::cerr << "Warning: the output is the input file itself; the original image "
                  << outputPath << " will be replaced\n";
    } else {
        std::cerr << "Warning: " << outputPath << " already exists and will be overwritten\n";
    }
}

// Makes sure the folder that will hold `outputPath` exists, creating it (and any
// missing parent folders) if needed. Returns false and prints the reason on failure.
static bool ensureOutputFolder(const std::string& outputPath) {
    const std::filesystem::path folder = std::filesystem::path(outputPath).parent_path();
    if (folder.empty()) {
        return true;  // plain file name: the image goes in the current folder
    }

    // The error_code overload reports failures through `error` instead of throwing.
    // It returns true only if it actually created something, and does nothing
    // (without an error) when the folder already exists.
    std::error_code error;
    const bool created = std::filesystem::create_directories(folder, error);
    if (error) {
        std::cerr << "Could not create output folder " << folder << ": " << error.message() << "\n";
        return false;
    }
    if (created) {
        // Say so, in case the folder name was a typo.
        std::cout << "Created output folder " << folder << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    // Energy mode: per-channel unless the user asks for grayscale.
    sc::EnergyMode energyMode = sc::EnergyMode::PerChannel;
    std::string energyName = "rgb";
    if (argc == 6) {
        energyName = argv[5];
        if (energyName == "gray") {
            energyMode = sc::EnergyMode::Grayscale;
        } else if (energyName != "rgb") {
            std::cerr << "Unknown energy mode: " << energyName << "\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // Reject an output name OpenCV cannot write (missing or unknown extension) before
    // doing any work, so a bad name never costs a full run or leaves a new folder behind.
    if (!cv::haveImageWriter(outputPath)) {
        std::cerr << "Unsupported output image name: " << outputPath
                  << " (use an extension such as .png or .jpg)\n";
        return EXIT_FAILURE;
    }

    const cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);  // always 8-bit BGR
    if (image.empty()) {
        std::cerr << "Could not read image: " << inputPath << "\n";
        return EXIT_FAILURE;
    }

    // Warn before the (possibly long) carving starts, so the run can still be cancelled.
    warnIfOverwriting(inputPath, outputPath);

    try {
        const int targetWidth = std::stoi(argv[3]);
        const int targetHeight = std::stoi(argv[4]);

        const auto start = std::chrono::steady_clock::now();
        const cv::Mat result = sc::seamCarve(image, targetWidth, targetHeight,
                                             sc::findVerticalSeamDP, energyMode);
        const auto stop = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        // Create the folder only now, after everything else has succeeded,
        // so a failed run does not leave an empty folder behind.
        if (!ensureOutputFolder(outputPath)) {
            return EXIT_FAILURE;
        }
        if (!cv::imwrite(outputPath, result)) {
            std::cerr << "Could not write image: " << outputPath << "\n";
            return EXIT_FAILURE;
        }
        std::cout << image.cols << "x" << image.rows << " -> " << result.cols << "x" << result.rows
                  << " in " << ms << " ms (energy: " << energyName << ")\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
