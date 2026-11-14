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
//   frames  also save one picture per seam, showing the resize step by step

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
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
              << " [dp|greedy|graphcut|graphcut-dinic] [rgb|gray] [seams] [frames]\n"
              << "  <image name>  an image inside the " << kInputFolder << " folder, e.g. castle.png\n"
              << "  dp      find seams by dynamic programming (default)\n"
              << "  greedy  find seams by the greedy row-by-row rule\n"
              << "  graphcut  find seams by minimum cut (optimal like dp, but much slower)\n"
              << "  graphcut-dinic  the same minimum cut with Dinic's faster max-flow algorithm\n"
              << "  rgb     energy from each colour channel, summed (default)\n"
              << "  gray    energy from the grayscale image\n"
              << "  seams   also save the original image with the removed seams painted on it\n"
              << "          (vertical seams red, horizontal seams yellow)\n"
              << "  frames  also save one picture per seam in a folder, showing each seam just\n"
              << "          before it is removed (a step-by-step view of the resize)\n"
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

// Makes sure `folder` exists, creating it (and any missing parent folders) if needed.
// Returns false and prints the reason on failure.
static bool ensureFolder(const fs::path& folder) {
    // The error_code overload reports failures through `error` instead of throwing.
    // It returns true only if it actually created something, and does nothing
    // (without an error) when the folder already exists.
    std::error_code error;
    const bool created = fs::create_directories(folder, error);
    if (error) {
        std::cerr << "Could not create folder " << folder << ": " << error.message() << "\n";
        return false;
    }
    if (created) {
        std::cout << "Created folder " << folder << "\n";
    }
    return true;
}

// Prints a warning if the frames folder already holds files from an earlier run.
static void warnIfFramesExist(const fs::path& framesFolder) {
    std::error_code error;
    if (fs::is_directory(framesFolder, error) && !fs::is_empty(framesFolder, error)) {
        std::cerr << "Warning: " << framesFolder.string()
                  << " already has frames in it; they will be overwritten\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 8) {  // three required arguments plus at most four options
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
    bool saveFrames = false;

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
        } else if (option == "frames") {
            saveFrames = true;
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
    const fs::path framesFolder = kOutputFolder / (outputStem + "_frames");

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
    if (saveFrames) {
        warnIfFramesExist(framesFolder);
    }

    // Step-by-step frames. saveFrame writes the next numbered picture: frame_0001.png,
    // frame_0002.png, ... The folder is created when the first frame is written, which is
    // after the target size has been checked, so an invalid run leaves no folder behind.
    int framesSaved = 0;
    const auto saveFrame = [&](const cv::Mat& frame) {
        if (framesSaved == 0 && !ensureFolder(framesFolder)) {
            throw std::runtime_error("could not create the frames folder");
        }
        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << (framesSaved + 1) << ".png";
        const fs::path framePath = framesFolder / name.str();
        if (!cv::imwrite(framePath.string(), frame)) {
            throw std::runtime_error("could not write " + framePath.string());
        }
        ++framesSaved;
    };

    // The observer is called by the resize loop once per seam, just before the seam is
    // removed. Every frame is drawn on a canvas the size of the original image, so all
    // frames have the same size. Left empty (no observer) unless frames were asked for.
    sc::SeamObserver frameObserver = nullptr;
    if (saveFrames) {
        frameObserver = [&](const cv::Mat& current, const sc::Seam& seam, bool horizontal) {
            saveFrame(sc::drawSeamFrame(current, seam, horizontal, image.size()));
        };
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        // With the seams option, the same resize also paints each removed seam onto `overlay`.
        // Seams and frames both add work to the run, so leave them off when timing.
        sc::CarveStats stats;
        cv::Mat overlay;
        const cv::Mat result = saveSeams
            ? sc::seamCarveWithOverlay(image, targetWidth, targetHeight, overlay,
                                       findSeam, energyMode, &stats, frameObserver)
            : sc::seamCarve(image, targetWidth, targetHeight, findSeam, energyMode, &stats,
                            frameObserver);
        const auto stop = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        // Create the folder only now, after everything else has succeeded,
        // so a failed run does not leave an empty folder behind.
        if (!ensureFolder(kOutputFolder)) {
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

        if (saveFrames) {
            // One last frame with nothing painted, so the sequence ends on the result.
            saveFrame(sc::drawSeamFrame(result, sc::Seam{}, false, image.size()));
            std::cout << "Saved " << framesSaved << (framesSaved == 1 ? " frame in " : " frames in ")
                      << framesFolder.string() << " (one per seam, then the final image)\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
