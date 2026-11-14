#pragma once

#include <vector>

namespace sc {

// Which maximum-flow algorithm FlowNetwork::maxFlow runs. Both return the same flow
// value and lead to the same minimum cut; they differ only in how fast they get there.
enum class MaxFlowAlgorithm {
    EdmondsKarp,  // one shortest augmenting path per BFS; O(V * E^2). Suggested by the brief.
    Dinic         // many augmenting paths per BFS (a "blocking flow"); O(V^2 * E)
};

// A directed graph with edge capacities, plus two maximum-flow algorithms.
// It knows nothing about images: the graph cut seam finder builds one of these from an
// energy map, and the unit tests build small textbook networks.
//
// Usage: add the edges, call maxFlow(source, sink) once, then sourceSide(source) gives
// the minimum cut.
class FlowNetwork {
public:
    explicit FlowNetwork(int nodeCount);

    // Adds a directed edge `from -> to`. Behind the scenes it also adds the reverse edge
    // `to -> from` with capacity 0, which the algorithm uses to undo flow it pushed earlier.
    void addEdge(int from, int to, double capacity);

    // Pushes as much flow as possible from source to sink and returns the total, which
    // by the max-flow min-cut theorem equals the capacity of the minimum cut.
    double maxFlow(int source, int sink,
                   MaxFlowAlgorithm algorithm = MaxFlowAlgorithm::EdmondsKarp);

    // After maxFlow: result[v] is true if node v can still be reached from the source
    // through edges with spare capacity. Those nodes are the source side S of a minimum
    // cut; every edge leaving S is full.
    std::vector<bool> sourceSide(int source) const;

    int nodeCount() const { return static_cast<int>(adjacency_.size()); }

private:
    // Edmonds-Karp: repeatedly find a shortest path (fewest edges, by BFS) from source to
    // sink that still has spare capacity, and push as much flow along it as it can take.
    // Stops when no such path is left. Every path costs one full BFS.
    double maxFlowEdmondsKarp(int source, int sink);

    // Dinic: one BFS labels every node with its distance from the source (its "level").
    // Then, without another BFS, push flow along as many paths as possible that go
    // strictly one level deeper at each step, until no such path is left. Repeat with a
    // new BFS. The sink gets further away after every round, so there are at most V rounds.
    double maxFlowDinic(int source, int sink);

    struct Edge {
        int to;           // node this edge points at
        double residual;  // capacity still unused (capacity minus the flow pushed so far)
    };

    // Edges are stored in pairs: edge 2k is a real edge and edge 2k+1 is its reverse.
    // So the partner of edge e is always e ^ 1 (flip the last bit).
    std::vector<Edge> edges_;
    std::vector<std::vector<int>> adjacency_;  // adjacency_[v] = indices of edges leaving v
};

}  // namespace sc
