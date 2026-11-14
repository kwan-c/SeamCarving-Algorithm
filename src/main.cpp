// Usage: seam_carve <image name> <target width> <target height> [options]
//
// Run it from the project folder. The image is read from the "images" folder and the
// results are always written to "images/out", with a file name built from the settings:
//
//   images/castle.png  ->  images/out/castle_186x140_dp_rgb.png
//
// Options, in any order after the three required arguments:
//   dp      find seams by dynamic programming (default)
//   greedy  find seams by the greedy row-by-row rule
//   graphcut  find seams by minimum cut / maximum flow (optimal like dp, but much slower)
//   graphcut-dinic  the same minimum cut, computed with Dinic's faster max-flow algorithm
//   rgb     energy from Sobel on each colour channel, summed (default)
//   gray    energy from Sobel on the grayscale image
//   seams   also save the original image with every removed seam painted on it

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>

#include <opencv2/imgcodecs.hpp>

#include "seam_carving.hpp"
#include "seam_overlay.hpp"

namespace fs = std::filesystem;

// Where images are read from and written to, relative to the folder the program is run in.
static const fs::path kInputFolder = "images";
static const fs::path kOutputFolder = kInputFolder / "out";

static void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <image name> <target width> <target height>"
              << " [dp|greedy|graphcut|graphcut-dinic] [rgb|gray] [seams]\n"
              << "  <image name>  an image inside the " << kInputFolder << " folder, e.g. castle.png\n"
              << "  dp      find seams by dynamic programming (default)\n"
              << "  greedy  find seams by the greedy row-by-row rule\n"
              << "  graphcut  find seams by minimum cut (optimal like dp, but much slower)\n"
              << "  graphcut-dinic  the same minimum cut with Dinic's faster max-flow algorithm\n"
              << "  rgb     energy from each colour channel, summed (default)\n"
              << "  gray    energy from the grayscale image\n"
              << "  seams   also save the original image with the removed seams painted on it\n"
              << "          (vertical seams red, horizontal seams yellow)\n"
              << "Run the program from the project folder. Results are written to " << kOutputFolder
              << ",\nnamed after the image and the settings, e.g. castle_186x140_dp_rgb.png.\n"
              << "An existing result with the same name is overwritten, with a warning.\n";
}

// Turns the first argument into the path of the image to read. The argument may be a
// plain file name ("castle.png") or the same name with the folder in front
// ("images/castle.png", which is what tab completion in a terminal produces).
// Anything in another folder is refused: returns false and prints why.
static bool resolveInputPath(const std::string& argument, fs::path& inputPath) {
    const fs::path given(argument);
    const fs::path folder = given.parent_path().lexically_normal();  // "./images" -> "images"
    if (given.filename().empty() || !(folder.empty() || folder == kInputFolder)) {
        std::cerr << "The image must be directly inside the " << kInputFolder
                  << " folder. Copy it there and give its file name, e.g. castle.png\n";
        return false;
    }
    inputPath = kInputFolder / given.filename();
    return true;
}

// Reads a whole number such as "186". Returns false for anything else ("abc", "12.5", "40px").
static bool parseWholeNumber(const std::string& text, int& value) {
    try {
        size_t charactersUsed = 0;
        value = std::stoi(text, &charactersUsed);
        return charactersUsed == text.size();  // false if something was left over
    } catch (const std::exception&) {
        return false;  // not a number at all, or too large for an int
    }
}

// Prints a warning if writing to `outputPath` would replace an existing file.
// It only warns; the run continues, and the file is replaced once the run succeeds.
static void warnIfOverwriting(const fs::path& outputPath) {
    // The error_code overload returns false instead of throwing if the path cannot be checked.
    std::error_code error;
    if (fs::is_regular_file(outputPath, error)) {
        std::cerr << "Warning: " << outputPath.string() << " already exists and will be overwritten\n";
    }
}

// Makes sure the output folder exists, creating it (and any missing parent folders)
// if needed. Returns false and prints the reason on failure.
static bool ensureOutputFolder() {
    // The error_code overload reports failures through `error` instead of throwing.
    // It returns true only if it actually created something, and does nothing
    // (without an error) when the folder already exists.
    std::error_code error;
    const bool created = fs::create_directories(kOutputFolder, error);
    if (error) {
        std::cerr << "Could not create output folder " << kOutputFolder << ": "
                  << error.message() << "\n";
        return false;
    }
    if (created) {
        std::cout << "Created output folder " << kOutputFolder << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 7) {  // three required arguments plus at most three options
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    fs::path inputPath;
    if (!resolveInputPath(argv[1], inputPath)) {
        return EXIT_FAILURE;
    }

    int targetWidth = 0, targetHeight = 0;
    if (!parseWholeNumber(argv[2], targetWidth) || !parseWholeNumber(argv[3], targetHeight)) {
        std::cerr << "The target width and height must be whole numbers, e.g. 186 140\n";
        return EXIT_FAILURE;
    }

    // Defaults: dynamic programming with per-channel energy, no seam overlay.
    sc::SeamFinder findSeam = sc::findVerticalSeamDP;
    std::string algorithmName = "dp";
    sc::EnergyMode energyMode = sc::EnergyMode::PerChannel;
    std::string energyName = "rgb";
    bool saveSeams = false;

    // Every argument after the first three is an option keyword, in any order.
    for (int i = 4; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "dp") {
            findSeam = sc::findVerticalSeamDP;
            algorithmName = option;
        } else if (option == "greedy") {
            findSeam = sc::findVerticalSeamGreedy;
            algorithmName = option;
        } else if (option == "graphcut") {
            findSeam = sc::findVerticalSeamGraphCut;
            algorithmName = option;
        } else if (option == "graphcut-dinic") {
            findSeam = sc::findVerticalSeamGraphCutDinic;
            algorithmName = option;
        } else if (option == "rgb") {
            energyMode = sc::EnergyMode::PerChannel;
            energyName = option;
        } else if (option == "gray") {
            energyMode = sc::EnergyMode::Grayscale;
            energyName = option;
        } else if (option == "seams") {
            saveSeams = true;
        } else {
            std::cerr << "Unknown option: " << option << "\n";
            if (option.compare(0, 6, "seams=") == 0) {
                std::cerr << "(The seam image no longer takes a file name. Write just: seams)\n";
            }
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    const cv::Mat image = cv::imread(inputPath.string(), cv::IMREAD_COLOR);  // always 8-bit BGR
    if (image.empty()) {
        std::cerr << "Could not read image: " << inputPath.string() << "\n"
                  << "Put the image in the " << kInputFolder
                  << " folder and run the program from the project folder.\n";
        return EXIT_FAILURE;
    }

    // Output names are built from the image name and the settings, so runs with different
    // settings never overwrite each other: castle_186x140_dp_rgb.png. They are always
    // PNG, which stores the pixels exactly (JPEG would blur the painted seams).
    const std::string outputStem = inputPath.stem().string() + "_" +
                                   std::to_string(targetWidth) + "x" + std::to_string(targetHeight) +
                                   "_" + algorithmName + "_" + energyName;
    const fs::path outputPath = kOutputFolder / (outputStem + ".png");
    const fs::path seamsPath = kOutputFolder / (outputStem + "_seams.png");

    if (algorithmName == "graphcut" || algorithmName == "graphcut-dinic") {
        std::cerr << "Note: graph cut solves a maximum-flow problem for every seam and is far "
                     "slower than dp. On large images it can take seconds per seam; "
                     "graphcut-dinic is the faster of the two.\n";
    }

    // Warn before the (possibly long) carving starts, so the run can still be cancelled.
    warnIfOverwriting(outputPath);
    if (saveSeams) {
        warnIfOverwriting(seamsPath);
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        // With the seams option, the same resize also paints each removed seam onto `overlay`.
        // The bookkeeping for that adds a little time, so leave it off when timing.
        sc::CarveStats stats;
        cv::Mat overlay;
        const cv::Mat result = saveSeams
            ? sc::seamCarveWithOverlay(image, targetWidth, targetHeight, overlay,
                                       findSeam, energyMode, &stats)
            : sc::seamCarve(image, targetWidth, targetHeight, findSeam, energyMode, &stats);
        const auto stop = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        // Create the folder only now, after everything else has succeeded,
        // so a failed run does not leave an empty folder behind.
        if (!ensureOutputFolder()) {
            return EXIT_FAILURE;
        }
        if (!cv::imwrite(outputPath.string(), result)) {
            std::cerr << "Could not write image: " << outputPath.string() << "\n";
            return EXIT_FAILURE;
        }
        std::cout << image.cols << "x" << image.rows << " -> " << result.cols << "x" << result.rows
                  << " in " << ms << " ms (algorithm: " << algorithmName
                  << ", energy: " << energyName << ")\n"
                  << stats.seamsRemoved << " seams removed, total seam energy "
                  << std::fixed << std::setprecision(0) << stats.totalSeamEnergy << "\n"
                  << "Saved " << outputPath.string() << "\n";

        if (saveSeams) {
            if (!cv::imwrite(seamsPath.string(), overlay)) {
                std::cerr << "Could not write image: " << seamsPath.string() << "\n";
                return EXIT_FAILURE;
            }
            std::cout << "Saved " << seamsPath.string()
                      << " (removed seams: vertical red, horizontal yellow)\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
