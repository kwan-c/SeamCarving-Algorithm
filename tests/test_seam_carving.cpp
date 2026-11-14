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

#include "max_flow.hpp"
#include "seam_carving.hpp"
#include "seam_overlay.hpp"

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

// A small noisy colour image: every pixel differs from its neighbours, so no seam is free.
static cv::Mat makeNoiseImage() {
    cv::Mat image(20, 30, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return image;
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

static const sc::MaxFlowAlgorithm bothMaxFlowAlgorithms[] = {
    sc::MaxFlowAlgorithm::EdmondsKarp, sc::MaxFlowAlgorithm::Dinic};

// The flow network from CLRS (Introduction to Algorithms), whose maximum flow is 23.
// Nodes: 0 = s, 1..4 = v1..v4, 5 = t. Checked with both max-flow algorithms.
static void testMaxFlowTextbookNetwork() {
    for (sc::MaxFlowAlgorithm algorithm : bothMaxFlowAlgorithms) {
        sc::FlowNetwork network(6);
        network.addEdge(0, 1, 16);
        network.addEdge(0, 2, 13);
        network.addEdge(1, 3, 12);
        network.addEdge(2, 1, 4);
        network.addEdge(2, 4, 14);
        network.addEdge(3, 2, 9);
        network.addEdge(3, 5, 20);
        network.addEdge(4, 3, 7);
        network.addEdge(4, 5, 4);
        CHECK(network.maxFlow(0, 5, algorithm) == 23.0);

        // The minimum cut separates {s, v1, v2, v4} from {v3, t}.
        // The edges leaving that set are v1->v3 (12), v4->v3 (7) and v4->t (4): 23 in total.
        const std::vector<bool> side = network.sourceSide(0);
        CHECK((side == std::vector<bool>{true, true, true, false, true, false}));
    }
}

static void testMaxFlowSimpleCases() {
    for (sc::MaxFlowAlgorithm algorithm : bothMaxFlowAlgorithms) {
        // A single edge carries exactly its capacity.
        sc::FlowNetwork single(2);
        single.addEdge(0, 1, 5);
        CHECK(single.maxFlow(0, 1, algorithm) == 5.0);

        // A chain is limited by its narrowest edge.
        sc::FlowNetwork chain(4);
        chain.addEdge(0, 1, 7);
        chain.addEdge(1, 2, 3);
        chain.addEdge(2, 3, 9);
        CHECK(chain.maxFlow(0, 3, algorithm) == 3.0);
        CHECK((chain.sourceSide(0) == std::vector<bool>{true, true, false, false}));

        // Two separate routes add up, and so do two parallel edges.
        sc::FlowNetwork parallel(4);
        parallel.addEdge(0, 1, 4);
        parallel.addEdge(1, 3, 4);
        parallel.addEdge(0, 2, 6);
        parallel.addEdge(2, 3, 5);
        parallel.addEdge(0, 3, 1);
        parallel.addEdge(0, 3, 2);
        CHECK(parallel.maxFlow(0, 3, algorithm) == 4.0 + 5.0 + 1.0 + 2.0);

        // The classic case that needs a reverse edge: a path may use the middle edge
        // 1->2 first, and the algorithm must be able to send that flow back.
        sc::FlowNetwork diamond(4);
        diamond.addEdge(0, 1, 1);
        diamond.addEdge(0, 2, 1);
        diamond.addEdge(1, 2, 1);
        diamond.addEdge(1, 3, 1);
        diamond.addEdge(2, 3, 1);
        CHECK(diamond.maxFlow(0, 3, algorithm) == 2.0);

        // No route from source to sink: zero flow, and the sink is not on the source side.
        sc::FlowNetwork disconnected(3);
        disconnected.addEdge(0, 1, 5);
        CHECK(disconnected.maxFlow(0, 2, algorithm) == 0.0);
        CHECK((disconnected.sourceSide(0) == std::vector<bool>{true, true, false}));

        // Source and sink the same node: nothing to push.
        CHECK(single.maxFlow(1, 1, algorithm) == 0.0);
    }

    // Invalid edges are rejected.
    sc::FlowNetwork network(2);
    bool threwRange = false, threwNegative = false;
    try { network.addEdge(0, 7, 1); } catch (const std::invalid_argument&) { threwRange = true; }
    try { network.addEdge(0, 1, -1); } catch (const std::invalid_argument&) { threwNegative = true; }
    CHECK(threwRange && threwNegative);
}

// Random networks: the two algorithms must agree on the flow value and on the minimum
// cut. (The set of nodes still reachable from the source is the same for every maximum
// flow, so the two source sides must be identical, not merely equally cheap.)
static void testMaxFlowDinicMatchesEdmondsKarp() {
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> nodesDist(2, 12), capacity(0, 20);
    for (int trial = 0; trial < 500; ++trial) {
        const int nodes = nodesDist(rng);
        std::uniform_int_distribution<int> pick(0, nodes - 1);
        std::uniform_int_distribution<int> edgeCountDist(0, nodes * 3);
        const int edgeCount = edgeCountDist(rng);

        sc::FlowNetwork forEdmondsKarp(nodes), forDinic(nodes);
        for (int e = 0; e < edgeCount; ++e) {
            const int from = pick(rng), to = pick(rng), cap = capacity(rng);
            forEdmondsKarp.addEdge(from, to, cap);
            forDinic.addEdge(from, to, cap);
        }
        const int source = 0, sink = nodes - 1;
        CHECK(forEdmondsKarp.maxFlow(source, sink, sc::MaxFlowAlgorithm::EdmondsKarp) ==
              forDinic.maxFlow(source, sink, sc::MaxFlowAlgorithm::Dinic));
        CHECK(forEdmondsKarp.sourceSide(source) == forDinic.sourceSide(source));
    }
}

// Swapping the max-flow algorithm must not change the seam at all.
static void testGraphCutDinicSameSeam() {
    // The hand-worked example: cost 4 through the last column.
    const cv::Mat example = makeEnergy({{1, 2, 9},
                                        {9, 9, 1},
                                        {9, 9, 1}});
    CHECK((sc::findVerticalSeamGraphCutDinic(example) == sc::Seam{1, 2, 2}));

    std::mt19937 rng(31337);
    std::uniform_int_distribution<int> size(1, 12);
    std::uniform_int_distribution<int> value(0, 9);
    for (int trial = 0; trial < 500; ++trial) {
        cv::Mat energy(size(rng), size(rng), CV_64F);
        for (int i = 0; i < energy.rows; ++i)
            for (int j = 0; j < energy.cols; ++j)
                energy.at<double>(i, j) = value(rng);

        const sc::Seam dinic = sc::findVerticalSeamGraphCutDinic(energy);
        CHECK(isValidSeam(dinic, energy.rows, energy.cols));
        CHECK(dinic == sc::findVerticalSeamGraphCut(energy));  // identical seam
        CHECK(sc::seamCost(energy, dinic) == sc::seamCost(energy, sc::findVerticalSeamDP(energy)));
    }

    // And on real energy maps, including an all-zero one.
    const cv::Mat imageEnergy = sc::computeEnergy(makeNoiseImage());
    CHECK(sc::findVerticalSeamGraphCutDinic(imageEnergy) == sc::findVerticalSeamGraphCut(imageEnergy));
    const cv::Mat zero = cv::Mat::zeros(4, 6, CV_64F);
    CHECK(sc::findVerticalSeamGraphCutDinic(zero) == sc::findVerticalSeamGraphCut(zero));
    CHECK((sc::findVerticalSeamGraphCutDinic(makeEnergy({{5}, {7}, {3}})) == sc::Seam{0, 0, 0}));
}

// The same matrix as the DP hand-worked example. Graph cut is optimal, so it must
// find the cost-4 seam, which runs through the last column in rows 1 and 2.
static void testGraphCutHandWorkedExample() {
    const cv::Mat energy = makeEnergy({{1, 2, 9},
                                       {9, 9, 1},
                                       {9, 9, 1}});
    const sc::Seam seam = sc::findVerticalSeamGraphCut(energy);
    CHECK((seam == sc::Seam{1, 2, 2}));
    CHECK(sc::seamCost(energy, seam) == 4.0);
}

// Graph cut and DP are both optimal, so on any input their seams must cost the same.
// The seams themselves may differ when several seams tie, so only the cost is compared.
static void testGraphCutMatchesDPCost() {
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> size(1, 10);
    std::uniform_int_distribution<int> value(0, 9);
    for (int trial = 0; trial < 500; ++trial) {
        cv::Mat energy(size(rng), size(rng), CV_64F);
        for (int i = 0; i < energy.rows; ++i)
            for (int j = 0; j < energy.cols; ++j)
                energy.at<double>(i, j) = value(rng);

        const sc::Seam cut = sc::findVerticalSeamGraphCut(energy);
        CHECK(isValidSeam(cut, energy.rows, energy.cols));
        CHECK(sc::seamCost(energy, cut) == sc::seamCost(energy, sc::findVerticalSeamDP(energy)));
    }

    // The same on a real energy map: Sobel energy of a noisy colour image.
    const cv::Mat imageEnergy = sc::computeEnergy(makeNoiseImage());
    const sc::Seam cut = sc::findVerticalSeamGraphCut(imageEnergy);
    CHECK(isValidSeam(cut, imageEnergy.rows, imageEnergy.cols));
    CHECK(sc::seamCost(imageEnergy, cut) ==
          sc::seamCost(imageEnergy, sc::findVerticalSeamDP(imageEnergy)));
}

static void testGraphCutEdgeCases() {
    // Single column: the only possible seam.
    CHECK((sc::findVerticalSeamGraphCut(makeEnergy({{5}, {7}, {3}})) == sc::Seam{0, 0, 0}));

    // Single row: the smallest pixel, including when it is in the last column.
    CHECK((sc::findVerticalSeamGraphCut(makeEnergy({{4, 2, 8, 1, 6}})) == sc::Seam{3}));
    CHECK((sc::findVerticalSeamGraphCut(makeEnergy({{4, 2, 8, 6, 1}})) == sc::Seam{4}));

    // Two columns, where the best seam zigzags between them.
    CHECK((sc::findVerticalSeamGraphCut(makeEnergy({{1, 9}, {9, 1}, {1, 9}})) == sc::Seam{0, 1, 0}));

    // Uniform energy: every seam ties, so only validity and cost are checked.
    const cv::Mat flat = cv::Mat::ones(6, 5, CV_64F);
    const sc::Seam flatSeam = sc::findVerticalSeamGraphCut(flat);
    CHECK(isValidSeam(flatSeam, 6, 5));
    CHECK(sc::seamCost(flat, flatSeam) == 6.0);

    // All-zero energy: no flow at all can be pushed, and the seam must still be valid.
    const cv::Mat zero = cv::Mat::zeros(4, 6, CV_64F);
    CHECK(isValidSeam(sc::findVerticalSeamGraphCut(zero), 4, 6));
}

// The network exactly as the assignment brief describes it cannot remove a last-column
// pixel. Brute force over the seams that avoid the last column gives what it should find.
static void testGraphCutAsDescribedSkipsLastColumn() {
    // On the hand-worked example the true best seam (cost 4) needs the last column,
    // so the as-described network is forced onto a much worse seam.
    const cv::Mat energy = makeEnergy({{1, 2, 9},
                                       {9, 9, 1},
                                       {9, 9, 1}});
    const sc::Seam described = sc::findVerticalSeamGraphCutAsDescribed(energy);
    CHECK(isValidSeam(described, 3, 3));
    CHECK(sc::seamCost(energy, described) == 19.0);
    for (int col : described) CHECK(col != 2);

    // In general it equals the best seam of the image without its last column,
    // and is therefore never cheaper than the true minimum.
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> rowsDist(1, 6), colsDist(2, 6), value(0, 9);
    int strictlyWorse = 0;
    for (int trial = 0; trial < 300; ++trial) {
        cv::Mat full(rowsDist(rng), colsDist(rng), CV_64F);
        for (int i = 0; i < full.rows; ++i)
            for (int j = 0; j < full.cols; ++j)
                full.at<double>(i, j) = value(rng);

        const sc::Seam seam = sc::findVerticalSeamGraphCutAsDescribed(full);
        const double cost = sc::seamCost(full, seam);
        const cv::Mat withoutLast = full.colRange(0, full.cols - 1).clone();
        CHECK(isValidSeam(seam, withoutLast.rows, withoutLast.cols));  // never the last column
        CHECK(cost == bruteForceMinSeamCost(withoutLast));
        const double best = bruteForceMinSeamCost(full);
        CHECK(cost >= best);
        if (cost > best) ++strictlyWorse;
    }
    CHECK(strictlyWorse > 0);

    // A single column leaves it nothing it is allowed to remove.
    bool threw = false;
    try { sc::findVerticalSeamGraphCutAsDescribed(makeEnergy({{5}, {7}})); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
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

    // So does the graph cut finder, with either max-flow algorithm; because the two
    // always find the same seam, the whole resized image must come out identical.
    const cv::Mat dinicResult = sc::seamCarve(image, 17, 9, sc::findVerticalSeamGraphCutDinic);
    const cv::Mat graphCutResult = sc::seamCarve(image, 17, 9, sc::findVerticalSeamGraphCut);
    CHECK(graphCutResult.cols == 17 && graphCutResult.rows == 9 &&
          graphCutResult.type() == image.type());
    CHECK(dinicResult.size() == graphCutResult.size() &&
          cv::norm(dinicResult, graphCutResult, cv::NORM_INF) == 0.0);

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

// The observer must be called once per seam, in order, with the image as it is at that moment.
static void testResizeObserverSeesEverySeam() {
    const cv::Mat image = makeNoiseImage();  // 20 rows x 30 columns
    int verticalCalls = 0, horizontalCalls = 0;
    bool shapesCorrect = true, seamsValid = true, verticalBeforeHorizontal = true;

    const sc::SeamObserver observer = [&](const cv::Mat& current, const sc::Seam& seam,
                                          bool horizontal) {
        if (!horizontal) {
            // Width pass: 20 rows, and one column fewer each time, starting from 30.
            if (horizontalCalls > 0) verticalBeforeHorizontal = false;
            if (current.rows != 20 || current.cols != 30 - verticalCalls) shapesCorrect = false;
            ++verticalCalls;
        } else {
            // Height pass on the transposed image: 17 rows (the new width), and one
            // column fewer each time, starting from 20 (the original height).
            if (current.rows != 17 || current.cols != 20 - horizontalCalls) shapesCorrect = false;
            ++horizontalCalls;
        }
        if (!isValidSeam(seam, current.rows, current.cols)) seamsValid = false;
    };

    sc::seamCarve(image, 17, 9, sc::findVerticalSeamDP, sc::EnergyMode::PerChannel, nullptr, observer);
    CHECK(verticalCalls == 13 && horizontalCalls == 11);
    CHECK(shapesCorrect && seamsValid && verticalBeforeHorizontal);
}

// A noisy image that contains neither of the two paint colours, so that
// "is this pixel painted?" can be answered just by looking at its colour.
static cv::Mat makePaintFreeImage() {
    cv::Mat image(20, 30, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(200, 200, 200));  // every channel below 200
    return image;
}

static int countColour(const cv::Mat& image, const cv::Vec3b& colour) {
    int count = 0;
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
            if (image.at<cv::Vec3b>(i, j) == colour) ++count;
    return count;
}

// Width only: 8 vertical seams, each taking exactly one pixel from every row.
static void testOverlayVerticalSeams() {
    const cv::Mat image = makePaintFreeImage();
    cv::Mat overlay;
    sc::seamCarveWithOverlay(image, 22, 20, overlay);

    CHECK(overlay.size() == image.size() && overlay.type() == CV_8UC3);  // original size
    CHECK(countColour(overlay, sc::kVerticalSeamColour) == 20 * 8);
    CHECK(countColour(overlay, sc::kHorizontalSeamColour) == 0);

    for (int i = 0; i < overlay.rows; ++i) {
        int paintedInRow = 0;
        for (int j = 0; j < overlay.cols; ++j) {
            if (overlay.at<cv::Vec3b>(i, j) == sc::kVerticalSeamColour) {
                ++paintedInRow;
            } else {
                // Every pixel that was not removed still shows the original picture.
                CHECK(overlay.at<cv::Vec3b>(i, j) == image.at<cv::Vec3b>(i, j));
            }
        }
        CHECK(paintedInRow == 8);
    }
}

// Width 30 -> 22 (8 vertical seams over 20 rows), then height 20 -> 13
// (7 horizontal seams over the 22 columns that are left).
static void testOverlayBothDirections() {
    const cv::Mat image = makePaintFreeImage();
    cv::Mat overlay;
    const cv::Mat resized = sc::seamCarveWithOverlay(image, 22, 13, overlay);

    const int red = countColour(overlay, sc::kVerticalSeamColour);
    const int yellow = countColour(overlay, sc::kHorizontalSeamColour);
    CHECK(red == 20 * 8);
    CHECK(yellow == 22 * 7);

    // Painted pixels are exactly the removed ones, so the rest add up to the new image.
    CHECK(overlay.rows * overlay.cols - red - yellow == resized.rows * resized.cols);

    // The same check with the greedy finder, to show the overlay works for any algorithm.
    cv::Mat greedyOverlay;
    sc::seamCarveWithOverlay(image, 22, 13, greedyOverlay, sc::findVerticalSeamGreedy);
    CHECK(countColour(greedyOverlay, sc::kVerticalSeamColour) == 20 * 8);
    CHECK(countColour(greedyOverlay, sc::kHorizontalSeamColour) == 22 * 7);
}

// Drawing the seams must not change which seams are removed.
static void testOverlayMatchesPlainResize() {
    const cv::Mat image = makePaintFreeImage();
    sc::CarveStats plainStats, overlayStats;
    const cv::Mat plain = sc::seamCarve(image, 19, 12, sc::findVerticalSeamDP,
                                        sc::EnergyMode::PerChannel, &plainStats);
    cv::Mat overlay;
    const cv::Mat withOverlay = sc::seamCarveWithOverlay(image, 19, 12, overlay,
                                                         sc::findVerticalSeamDP,
                                                         sc::EnergyMode::PerChannel, &overlayStats);
    CHECK(plain.size() == withOverlay.size());
    CHECK(cv::norm(plain, withOverlay, cv::NORM_INF) == 0.0);
    CHECK(plainStats.seamsRemoved == overlayStats.seamsRemoved);
    CHECK(plainStats.totalSeamEnergy == overlayStats.totalSeamEnergy);

    // Removing nothing paints nothing.
    cv::Mat untouched;
    sc::seamCarveWithOverlay(image, 30, 20, untouched);
    CHECK(cv::norm(untouched, image, cv::NORM_INF) == 0.0);
}

static void testOverlayInputChecks() {
    // A grayscale image is accepted, and its overlay is in colour so the paint shows.
    cv::Mat gray(10, 12, CV_8UC1);
    cv::randu(gray, cv::Scalar(0), cv::Scalar(200));
    cv::Mat overlay;
    const cv::Mat resized = sc::seamCarveWithOverlay(gray, 9, 10, overlay);
    CHECK(resized.cols == 9 && resized.type() == CV_8UC1);
    CHECK(overlay.type() == CV_8UC3 && overlay.size() == gray.size());
    CHECK(countColour(overlay, sc::kVerticalSeamColour) == 10 * 3);

    // Empty images, unsupported pixel types and invalid targets are rejected.
    bool threwEmpty = false, threwType = false, threwTarget = false;
    try { sc::seamCarveWithOverlay(cv::Mat(), 1, 1, overlay); }
    catch (const std::invalid_argument&) { threwEmpty = true; }
    try { sc::seamCarveWithOverlay(cv::Mat(5, 5, CV_32FC3, cv::Scalar(0, 0, 0)), 4, 5, overlay); }
    catch (const std::invalid_argument&) { threwType = true; }
    try { sc::seamCarveWithOverlay(makePaintFreeImage(), 31, 20, overlay); }
    catch (const std::invalid_argument&) { threwTarget = true; }
    CHECK(threwEmpty && threwType && threwTarget);
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
    {"maxflow_textbook_network",            testMaxFlowTextbookNetwork},
    {"maxflow_simple_cases",                testMaxFlowSimpleCases},
    {"maxflow_dinic_matches_edmonds_karp",  testMaxFlowDinicMatchesEdmondsKarp},
    {"graphcut_hand_worked_example",        testGraphCutHandWorkedExample},
    {"graphcut_matches_dp_cost",            testGraphCutMatchesDPCost},
    {"graphcut_edge_cases",                 testGraphCutEdgeCases},
    {"graphcut_as_described_skips_last_column", testGraphCutAsDescribedSkipsLastColumn},
    {"graphcut_dinic_same_seam",            testGraphCutDinicSameSeam},
    {"remove_vertical_seam",                testRemoveVerticalSeam},
    {"energy_both_modes",                   testComputeEnergy},
    {"energy_equal_brightness_colour_edge", testEqualBrightnessColourEdge},
    {"resize_reaches_target_size",          testResizeReachesTargetSize},
    {"resize_statistics",                   testResizeStatistics},
    {"resize_rejects_invalid_input",        testResizeRejectsInvalidInput},
    {"resize_observer_sees_every_seam",     testResizeObserverSeesEverySeam},
    {"overlay_vertical_seams",              testOverlayVerticalSeams},
    {"overlay_both_directions",             testOverlayBothDirections},
    {"overlay_matches_plain_resize",        testOverlayMatchesPlainResize},
    {"overlay_input_checks",                testOverlayInputChecks},
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
