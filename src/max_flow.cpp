#include "max_flow.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sc {

FlowNetwork::FlowNetwork(int nodeCount) {
    if (nodeCount < 0) {
        throw std::invalid_argument("FlowNetwork: node count cannot be negative");
    }
    adjacency_.resize(nodeCount);
}

void FlowNetwork::addEdge(int from, int to, double capacity) {
    if (from < 0 || from >= nodeCount() || to < 0 || to >= nodeCount()) {
        throw std::invalid_argument("FlowNetwork::addEdge: node index out of range");
    }
    if (capacity < 0.0) {
        throw std::invalid_argument("FlowNetwork::addEdge: capacity cannot be negative");
    }
    adjacency_[from].push_back(static_cast<int>(edges_.size()));
    edges_.push_back({to, capacity});  // the real edge
    adjacency_[to].push_back(static_cast<int>(edges_.size()));
    edges_.push_back({from, 0.0});     // its reverse, empty until flow is pushed forward
}

double FlowNetwork::maxFlow(int source, int sink, MaxFlowAlgorithm algorithm) {
    if (source < 0 || source >= nodeCount() || sink < 0 || sink >= nodeCount()) {
        throw std::invalid_argument("FlowNetwork::maxFlow: node index out of range");
    }
    if (source == sink) {
        return 0.0;
    }
    return algorithm == MaxFlowAlgorithm::Dinic ? maxFlowDinic(source, sink)
                                                : maxFlowEdmondsKarp(source, sink);
}

double FlowNetwork::maxFlowEdmondsKarp(int source, int sink) {
    double totalFlow = 0.0;
    std::vector<int> edgeInto(nodeCount());  // edgeInto[v] = edge the BFS used to reach v
    std::vector<int> queue;
    queue.reserve(nodeCount());

    while (true) {
        // --- BFS from the source over edges that still have spare capacity ---
        std::fill(edgeInto.begin(), edgeInto.end(), -1);  // -1 means "not reached yet"
        queue.clear();
        queue.push_back(source);
        bool reachedSink = false;
        for (size_t head = 0; head < queue.size() && !reachedSink; ++head) {
            const int node = queue[head];
            for (int edgeIndex : adjacency_[node]) {
                const Edge& edge = edges_[edgeIndex];
                if (edge.residual > 0.0 && edgeInto[edge.to] == -1 && edge.to != source) {
                    edgeInto[edge.to] = edgeIndex;
                    if (edge.to == sink) {
                        reachedSink = true;
                        break;
                    }
                    queue.push_back(edge.to);
                }
            }
        }
        if (!reachedSink) {
            break;  // no augmenting path is left, so the flow is maximum
        }

        // --- Walk back from the sink to find the tightest edge on the path ---
        // edges_[e ^ 1].to is the node that edge e starts from.
        double bottleneck = std::numeric_limits<double>::infinity();
        for (int node = sink; node != source; node = edges_[edgeInto[node] ^ 1].to) {
            bottleneck = std::min(bottleneck, edges_[edgeInto[node]].residual);
        }

        // --- Push that much flow along the path ---
        // Each edge loses spare capacity, and its reverse gains the same amount so that
        // a later path is allowed to send this flow back.
        for (int node = sink; node != source; node = edges_[edgeInto[node] ^ 1].to) {
            edges_[edgeInto[node]].residual -= bottleneck;
            edges_[edgeInto[node] ^ 1].residual += bottleneck;
        }
        totalFlow += bottleneck;
    }
    return totalFlow;
}

double FlowNetwork::maxFlowDinic(int source, int sink) {
    double totalFlow = 0.0;
    std::vector<int> level(nodeCount());     // BFS distance from the source, -1 if unreachable
    std::vector<size_t> nextEdge(nodeCount());  // per node: first edge not yet ruled out this round
    std::vector<int> queue;
    queue.reserve(nodeCount());
    std::vector<int> path;                   // edges of the path being built, source first

    while (true) {
        // --- BFS: label every node with its distance from the source ---
        std::fill(level.begin(), level.end(), -1);
        level[source] = 0;
        queue.clear();
        queue.push_back(source);
        for (size_t head = 0; head < queue.size(); ++head) {
            const int node = queue[head];
            for (int edgeIndex : adjacency_[node]) {
                const Edge& edge = edges_[edgeIndex];
                if (edge.residual > 0.0 && level[edge.to] == -1) {
                    level[edge.to] = level[node] + 1;
                    queue.push_back(edge.to);
                }
            }
        }
        if (level[sink] == -1) {
            break;  // the sink cannot be reached any more, so the flow is maximum
        }

        // --- Blocking flow: push along paths that go exactly one level deeper per edge ---
        // The walk has three moves: advance along a usable edge, retreat from a dead end,
        // and augment when it reaches the sink. nextEdge remembers which edges of a node
        // have already been ruled out, so no edge is ever looked at twice for nothing.
        std::fill(nextEdge.begin(), nextEdge.end(), 0);
        path.clear();
        int node = source;
        while (true) {
            if (node == sink) {
                // Augment: find the tightest edge on the path and push that much flow.
                double bottleneck = std::numeric_limits<double>::infinity();
                for (int edgeIndex : path) {
                    bottleneck = std::min(bottleneck, edges_[edgeIndex].residual);
                }
                for (int edgeIndex : path) {
                    edges_[edgeIndex].residual -= bottleneck;
                    edges_[edgeIndex ^ 1].residual += bottleneck;
                }
                totalFlow += bottleneck;
                path.clear();  // start the next path from the source again
                node = source;
            } else if (nextEdge[node] < adjacency_[node].size()) {
                const int edgeIndex = adjacency_[node][nextEdge[node]];
                const Edge& edge = edges_[edgeIndex];
                if (edge.residual > 0.0 && level[edge.to] == level[node] + 1) {
                    path.push_back(edgeIndex);  // advance
                    node = edge.to;
                } else {
                    ++nextEdge[node];           // this edge is no use in this round
                }
            } else {
                // Dead end: every edge out of this node has been ruled out.
                if (path.empty()) {
                    break;  // even the source is stuck, so this round is finished
                }
                const int edgeIndex = path.back();  // retreat along the edge we came by
                path.pop_back();
                node = edges_[edgeIndex ^ 1].to;    // the node that edge started from
                ++nextEdge[node];                   // and never try that edge again this round
            }
        }
    }
    return totalFlow;
}

std::vector<bool> FlowNetwork::sourceSide(int source) const {
    if (source < 0 || source >= nodeCount()) {
        throw std::invalid_argument("FlowNetwork::sourceSide: node index out of range");
    }
    std::vector<bool> reached(nodeCount(), false);
    std::vector<int> queue;
    queue.reserve(nodeCount());
    reached[source] = true;
    queue.push_back(source);
    for (size_t head = 0; head < queue.size(); ++head) {
        for (int edgeIndex : adjacency_[queue[head]]) {
            const Edge& edge = edges_[edgeIndex];
            if (edge.residual > 0.0 && !reached[edge.to]) {
                reached[edge.to] = true;
                queue.push_back(edge.to);
            }
        }
    }
    return reached;
}

}  // namespace sc
