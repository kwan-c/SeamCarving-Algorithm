// Unit tests with no test framework.
//
//   test_seam_carving            runs every test
//   test_seam_carving <name>     runs one test (this is how CTest calls it, once per test)
//   test_seam_carving --list     prints the test names
//
// Each CHECK prints a message on failure, and the program exits non-zero if any check
// failed (so CTest and CI can see it). We do not use assert() because Release builds
// compile it out.

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "seam_carving.hpp"

static int failures = 0;

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "FAILED: " #condition " (line " << __LINE__ << ")\n";    \
            ++failures;                                                           \
        }                                                                         \
    } while (false)

// Builds a CV_64F energy matrix from a nested list, so tests read like the report's tables.
static cv::Mat makeEnergy(const std::vector<std::vector<double>>& values) {
    cv::Mat energy(static_cast<int>(values.size()), static_cast<int>(values[0].size()), CV_64F);
    for (int i = 0; i < energy.rows; ++i)
        for (int j = 0; j < energy.cols; ++j)
            energy.at<double>(i, j) = values[i][j];
    return energy;
}

// One pixel per row, every column in range, neighbouring rows at most one column apart.
static bool isValidSeam(const sc::Seam& seam, int rows, int cols) {
    if (static_cast<int>(seam.size()) != rows) return false;
    for (int i = 0; i < rows; ++i) {
        if (seam[i] < 0 || seam[i] >= cols) return false;
        if (i > 0 && std::abs(seam[i] - seam[i - 1]) > 1) return false;
    }
    return true;
}

// Tries every possible seam (exponential time) and returns the cheapest cost.
// Only usable on tiny matrices; it is the ground truth the DP is compared against.
static double bruteForceCost(const cv::Mat& energy, int row, int col) {
    const double here = energy.at<double>(row, col);
    if (row == energy.rows - 1) return here;
    double best = std::numeric_limits<double>::infinity();
    for (int next = col - 1; next <= col + 1; ++next) {
        if (next < 0 || next >= energy.cols) continue;
        best = std::min(best, bruteForceCost(energy, row + 1, next));
    }
    return here + best;
}

static double bruteForceMinSeamCost(const cv::Mat& energy) {
    double best = std::numeric_limits<double>::infinity();
    for (int j = 0; j < energy.cols; ++j) best = std::min(best, bruteForceCost(energy, 0, j));
    return best;
}

// A worked example small enough to check by hand.
static void testHandWorkedExample() {
    const cv::Mat energy = makeEnergy({{1, 2, 9},
                                       {9, 9, 1},
                                       {9, 9, 1}});
    const sc::Seam seam = sc::findVerticalSeamDP(energy);
    CHECK((seam == sc::Seam{1, 2, 2}));          // 2 -> 1 -> 1
    CHECK(sc::seamCost(energy, seam) == 4.0);    // a greedy walk from the top would cost 19
}

// DP must match exhaustive search on many random small matrices.
// Integer energies make the comparison exact and produce plenty of ties.
static void testAgainstBruteForce() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> size(1, 6);
    std::uniform_int_distribution<int> value(0, 9);
    for (int trial = 0; trial < 1000; ++trial) {
        cv::Mat energy(size(rng), size(rng), CV_64F);
        for (int i = 0; i < energy.rows; ++i)
            for (int j = 0; j < energy.cols; ++j)
                energy.at<double>(i, j) = value(rng);

        const sc::Seam seam = sc::findVerticalSeamDP(energy);
        CHECK(isValidSeam(seam, energy.rows, energy.cols));
        CHECK(sc::seamCost(energy, seam) == bruteForceMinSeamCost(energy));
    }
}

static void testEdgeCases() {
    // Single column: the only possible seam is column 0 in every row.
    const sc::Seam singleColumn = sc::findVerticalSeamDP(makeEnergy({{5}, {7}, {3}}));
    CHECK((singleColumn == sc::Seam{0, 0, 0}));

    // Single row: the seam is just the smallest pixel.
    const sc::Seam singleRow = sc::findVerticalSeamDP(makeEnergy({{4, 2, 8, 1, 6}}));
    CHECK((singleRow == sc::Seam{3}));

    // Uniform energy: every seam costs the same, so only validity and cost are checked.
    const cv::Mat flat = cv::Mat::ones(6, 5, CV_64F);
    const sc::Seam flatSeam = sc::findVerticalSeamDP(flat);
    CHECK(isValidSeam(flatSeam, 6, 5));
    CHECK(sc::seamCost(flat, flatSeam) == 6.0);

    // Two columns: the border checks on both sides are exercised at once.
    const sc::Seam twoColumns = sc::findVerticalSeamDP(makeEnergy({{1, 9}, {9, 1}, {1, 9}}));
    CHECK((twoColumns == sc::Seam{0, 1, 0}));
}

// The same matrix as the DP hand-worked example, where greedy is far from optimal.
static void testGreedyHandWorkedExample() {
    const cv::Mat energy = makeEnergy({{1, 2, 9},
                                       {9, 9, 1},
                                       {9, 9, 1}});
    const sc::Seam seam = sc::findVerticalSeamGreedy(energy);
    // Starts at the 1 in column 0. Below it are 9 and 9: a tie, so it goes straight down.
    CHECK((seam == sc::Seam{0, 0, 0}));
    CHECK(sc::seamCost(energy, seam) == 19.0);   // the DP seam costs 4
}

// When the cheapest path is obvious at every single step, greedy finds the DP seam.
static void testGreedyMatchesDPOnEasyInput() {
    const cv::Mat energy = makeEnergy({{5, 1, 5, 5},
                                       {5, 5, 1, 5},
                                       {5, 5, 1, 5},
                                       {5, 1, 5, 5}});
    const sc::Seam greedy = sc::findVerticalSeamGreedy(energy);
    CHECK((greedy == sc::Seam{1, 2, 2, 1}));
    CHECK(greedy == sc::findVerticalSeamDP(energy));
}

// Greedy must always return a valid seam, and can never beat the DP because the DP is
// optimal. On random input it should also be strictly worse at least some of the time.
static void testGreedyAgainstDP() {
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> size(1, 12);
    std::uniform_int_distribution<int> value(0, 9);
    int strictlyWorse = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        cv::Mat energy(size(rng), size(rng), CV_64F);
        for (int i = 0; i < energy.rows; ++i)
            for (int j = 0; j < energy.cols; ++j)
                energy.at<double>(i, j) = value(rng);

        const sc::Seam greedy = sc::findVerticalSeamGreedy(energy);
        const double greedyCost = sc::seamCost(energy, greedy);
        const double dpCost = sc::seamCost(energy, sc::findVerticalSeamDP(energy));
        CHECK(isValidSeam(greedy, energy.rows, energy.cols));
        CHECK(greedyCost >= dpCost);
        if (greedyCost > dpCost) ++strictlyWorse;
    }
    CHECK(strictlyWorse > 0);
}

static void testGreedyEdgeCases() {
    // Single column: the only possible seam.
    CHECK((sc::findVerticalSeamGreedy(makeEnergy({{5}, {7}, {3}})) == sc::Seam{0, 0, 0}));

    // Single row: greedy and DP both just pick the smallest pixel.
    CHECK((sc::findVerticalSeamGreedy(makeEnergy({{4, 2, 8, 1, 6}})) == sc::Seam{3}));

    // Two columns: both border checks are exercised.
    CHECK((sc::findVerticalSeamGreedy(makeEnergy({{1, 9}, {9, 1}, {1, 9}})) == sc::Seam{0, 1, 0}));

    // Uniform energy: ties everywhere, so the seam starts at column 0 and goes straight down.
    const cv::Mat flat = cv::Mat::ones(6, 5, CV_64F);
    CHECK((sc::findVerticalSeamGreedy(flat) == sc::Seam{0, 0, 0, 0, 0, 0}));
}

// After removal, each row must be the original row with exactly the seam pixel missing.
static void testRemoveVerticalSeam() {
    cv::Mat image(3, 4, CV_8UC3);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(i), static_cast<uchar>(j), 200);

    const sc::Seam seam = {0, 1, 3};  // first column, a middle column, last column
    const cv::Mat out = sc::removeVerticalSeam(image, seam);
    CHECK(out.rows == 3 && out.cols == 3 && out.type() == image.type());

    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            const int originalColumn = (j < seam[i]) ? j : j + 1;
            CHECK(out.at<cv::Vec3b>(i, j) == image.at<cv::Vec3b>(i, originalColumn));
        }
    }
}

// Properties that must hold for both energy modes.
static void testComputeEnergy() {
    for (sc::EnergyMode mode : {sc::EnergyMode::PerChannel, sc::EnergyMode::Grayscale}) {
        // A flat image has no gradients anywhere.
        const cv::Mat flat(8, 10, CV_8UC3, cv::Scalar(120, 120, 120));
        const cv::Mat flatEnergy = sc::computeEnergy(flat, mode);
        CHECK(flatEnergy.type() == CV_64F && flatEnergy.size() == flat.size());
        CHECK(cv::countNonZero(flatEnergy) == 0);

        // A vertical black/white edge: energy is high next to the edge and zero far from it.
        cv::Mat edge(8, 10, CV_8UC3, cv::Scalar(0, 0, 0));
        edge.colRange(5, 10).setTo(cv::Scalar(255, 255, 255));
        const cv::Mat edgeEnergy = sc::computeEnergy(edge, mode);
        CHECK(edgeEnergy.type() == CV_64F);  // single channel, as the DP requires
        CHECK(edgeEnergy.at<double>(4, 4) > 0.0 && edgeEnergy.at<double>(4, 5) > 0.0);
        CHECK(edgeEnergy.at<double>(4, 1) == 0.0 && edgeEnergy.at<double>(4, 8) == 0.0);

        // So the best seam must avoid the two columns that touch the edge.
        const sc::Seam seam = sc::findVerticalSeamDP(edgeEnergy);
        for (int col : seam) CHECK(col != 4 && col != 5);
    }

    // The default mode is PerChannel.
    cv::Mat noise(8, 10, CV_8UC3);
    cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    CHECK(cv::norm(sc::computeEnergy(noise),
                   sc::computeEnergy(noise, sc::EnergyMode::PerChannel), cv::NORM_INF) == 0.0);

    // On an image that is already single-channel, the two modes are the same computation.
    cv::Mat gray(8, 10, CV_8UC1);
    cv::randu(gray, cv::Scalar(0), cv::Scalar(255));
    CHECK(cv::norm(sc::computeEnergy(gray, sc::EnergyMode::PerChannel),
                   sc::computeEnergy(gray, sc::EnergyMode::Grayscale), cv::NORM_INF) == 0.0);
}

// The case that separates the two modes: red next to a green of the same brightness.
// Both colours convert to gray value 76, so the edge is invisible in grayscale.
static void testEqualBrightnessColourEdge() {
    cv::Mat image(8, 10, CV_8UC3, cv::Scalar(0, 0, 255));   // red (OpenCV stores B, G, R)
    image.colRange(5, 10).setTo(cv::Scalar(0, 130, 0));     // green

    const cv::Mat grayEnergy = sc::computeEnergy(image, sc::EnergyMode::Grayscale);
    CHECK(cv::countNonZero(grayEnergy) == 0);               // grayscale sees a flat image

    const cv::Mat colourEnergy = sc::computeEnergy(image, sc::EnergyMode::PerChannel);
    CHECK(colourEnergy.at<double>(4, 4) > 0.0 && colourEnergy.at<double>(4, 5) > 0.0);
    CHECK(colourEnergy.at<double>(4, 1) == 0.0 && colourEnergy.at<double>(4, 8) == 0.0);

    // With per-channel energy the seam keeps away from the colour boundary.
    const sc::Seam seam = sc::findVerticalSeamDP(colourEnergy);
    for (int col : seam) CHECK(col != 4 && col != 5);
}

// A small noisy colour image: every pixel differs from its neighbours, so no seam is free.
static cv::Mat makeNoiseImage() {
    cv::Mat image(20, 30, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return image;
}

static void testResizeReachesTargetSize() {
    const cv::Mat image = makeNoiseImage();

    const cv::Mat narrower = sc::seamCarve(image, 22, 20);  // vertical seams only
    CHECK(narrower.cols == 22 && narrower.rows == 20);

    const cv::Mat shorter = sc::seamCarve(image, 30, 13);   // horizontal seams only
    CHECK(shorter.cols == 30 && shorter.rows == 13);

    const cv::Mat both = sc::seamCarve(image, 17, 9);
    CHECK(both.cols == 17 && both.rows == 9 && both.type() == image.type());

    // Asking for the current size removes nothing and leaves the pixels untouched.
    const cv::Mat same = sc::seamCarve(image, 30, 20);
    CHECK(cv::norm(same, image, cv::NORM_INF) == 0.0);

    // The grayscale alternative can be selected and reaches the same target size.
    const cv::Mat grayMode = sc::seamCarve(image, 17, 9, sc::findVerticalSeamDP,
                                           sc::EnergyMode::Grayscale);
    CHECK(grayMode.cols == 17 && grayMode.rows == 9 && grayMode.type() == image.type());

    // The greedy finder plugs into the same loop and reaches the same target size.
    const cv::Mat greedyResult = sc::seamCarve(image, 17, 9, sc::findVerticalSeamGreedy);
    CHECK(greedyResult.cols == 17 && greedyResult.rows == 9 && greedyResult.type() == image.type());

    // Carving all the way down to a single pixel must not crash.
    const cv::Mat tiny = sc::seamCarve(image, 1, 1);
    CHECK(tiny.cols == 1 && tiny.rows == 1);
}

static void testResizeStatistics() {
    const cv::Mat image = makeNoiseImage();

    // 30 -> 17 columns is 13 vertical seams; 20 -> 9 rows is 11 horizontal seams.
    sc::CarveStats dpStats, greedyStats;
    sc::seamCarve(image, 17, 9, sc::findVerticalSeamDP, sc::EnergyMode::PerChannel, &dpStats);
    sc::seamCarve(image, 17, 9, sc::findVerticalSeamGreedy, sc::EnergyMode::PerChannel, &greedyStats);
    CHECK(dpStats.seamsRemoved == 24 && greedyStats.seamsRemoved == 24);
    CHECK(dpStats.totalSeamEnergy > 0.0 && greedyStats.totalSeamEnergy > 0.0);

    // A stats struct that is reused starts again from zero.
    sc::seamCarve(image, 29, 20, sc::findVerticalSeamDP, sc::EnergyMode::PerChannel, &dpStats);
    CHECK(dpStats.seamsRemoved == 1);
}

// Invalid targets are rejected rather than silently ignored.
static void testResizeRejectsInvalidInput() {
    const cv::Mat image = makeNoiseImage();
    bool threwTooWide = false, threwZero = false, threwEmpty = false;
    try { sc::seamCarve(image, 31, 20); } catch (const std::invalid_argument&) { threwTooWide = true; }
    try { sc::seamCarve(image, 30, 0); }  catch (const std::invalid_argument&) { threwZero = true; }
    try { sc::seamCarve(cv::Mat(), 1, 1); } catch (const std::invalid_argument&) { threwEmpty = true; }
    CHECK(threwTooWide && threwZero && threwEmpty);
}

// ---------------------------------------------------------------------------
// Test list and runner
// ---------------------------------------------------------------------------
struct TestCase {
    const char* name;   // the name CTest shows; must also appear in CMakeLists.txt
    void (*run)();      // the function to call
};

static const std::vector<TestCase> allTests = {
    {"dp_hand_worked_example",              testHandWorkedExample},
    {"dp_matches_brute_force",              testAgainstBruteForce},
    {"dp_edge_cases",                       testEdgeCases},
    {"greedy_hand_worked_example",          testGreedyHandWorkedExample},
    {"greedy_matches_dp_on_easy_input",     testGreedyMatchesDPOnEasyInput},
    {"greedy_never_beats_dp",               testGreedyAgainstDP},
    {"greedy_edge_cases",                   testGreedyEdgeCases},
    {"remove_vertical_seam",                testRemoveVerticalSeam},
    {"energy_both_modes",                   testComputeEnergy},
    {"energy_equal_brightness_colour_edge", testEqualBrightnessColourEdge},
    {"resize_reaches_target_size",          testResizeReachesTargetSize},
    {"resize_statistics",                   testResizeStatistics},
    {"resize_rejects_invalid_input",        testResizeRejectsInvalidInput},
};

// Used by the CTest entry "test_list_is_complete": succeeds only if the names passed in
// (the list in CMakeLists.txt) are exactly the names in allTests. This catches a test
// that was added here but never registered with CTest, and the reverse.
static int checkNames(int argc, char** argv) {
    std::set<std::string> registered;
    for (int i = 2; i < argc; ++i) registered.insert(argv[i]);
    std::set<std::string> defined;
    for (const TestCase& test : allTests) defined.insert(test.name);

    bool same = true;
    for (const std::string& name : defined) {
        if (registered.count(name) == 0) {
            std::cerr << "Not registered in CMakeLists.txt: " << name << "\n";
            same = false;
        }
    }
    for (const std::string& name : registered) {
        if (defined.count(name) == 0) {
            std::cerr << "Registered in CMakeLists.txt but not defined here: " << name << "\n";
            same = false;
        }
    }
    return same ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
    const std::string wanted = (argc > 1) ? argv[1] : "";  // empty means "run everything"

    if (wanted == "--list") {
        for (const TestCase& test : allTests) std::cout << test.name << "\n";
        return EXIT_SUCCESS;
    }
    if (wanted == "--check-names") {
        return checkNames(argc, argv);
    }

    int testsRun = 0;
    for (const TestCase& test : allTests) {
        if (!wanted.empty() && wanted != test.name) continue;
        ++testsRun;
        const int failuresBefore = failures;
        try {
            test.run();
        } catch (const std::exception& error) {
            // An unexpected exception fails this test but lets the others still run.
            std::cerr << "FAILED: unexpected exception: " << error.what() << "\n";
            ++failures;
        }
        std::cout << (failures == failuresBefore ? "PASS  " : "FAIL  ") << test.name << "\n";
    }

    if (testsRun == 0) {
        std::cerr << "Unknown test: " << wanted << " (use --list to see the names)\n";
        return EXIT_FAILURE;
    }
    if (failures == 0) {
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
}
