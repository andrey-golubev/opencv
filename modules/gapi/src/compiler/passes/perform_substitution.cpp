// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#include <list>
#include <algorithm>

namespace cv { namespace gimpl {
namespace {
using Graph = GModel::Graph;

template<typename It, typename Callable>
void erase(Graph& g, It first, It last, Callable get) {
    for (; first != last; ++first) {
        ade::NodeHandle node = get(first);
        if (node == nullptr) continue;  // NB: some nodes might already be erased
        g.erase(node);
    }
}
}  // anonymous namespace

void performSubstitution(Graph& graph,
    const SubgraphMatch& patternToGraph, const SubgraphMatch& patternToSubstitute) {
    // substitute input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        // Note: we don't replace input DATA nodes here, only redirect their output edges
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);
        GModel::redirectReaders(graph, substituteDataNode, graphDataNode);
    }

    // substitute output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        // Note: we don't replace output DATA nodes here, only redirect their input edges
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);
        for (auto e : graphDataNode->inEdges()) {
            graph.erase(e);
        }
        GModel::redirectWriter(graph, substituteDataNode, graphDataNode);
    }

    // erase redundant nodes
    const auto get_from_node = [] (std::list<ade::NodeHandle>::const_iterator it) { return *it; };
    const auto get_from_pair = [] (SubgraphMatch::M::const_iterator it) { return it->second; };

    // erase input data nodes of __substitute__
    erase(graph, patternToSubstitute.inputDataNodes.begin(),
        patternToSubstitute.inputDataNodes.end(), get_from_pair);

    // erase old start OP nodes of __main graph__
    erase(graph, patternToGraph.startOpNodes.begin(),
        patternToGraph.startOpNodes.end(), get_from_pair);

    // erase old internal nodes of __main graph__
    erase(graph, patternToGraph.internalLayers.begin(),
        patternToGraph.internalLayers.end(), get_from_node);

    // erase old finish OP nodes of __main graph__
    erase(graph, patternToGraph.finishOpNodes.begin(),
        patternToGraph.finishOpNodes.end(), get_from_pair);

    // erase output data nodes of __substitute__
    erase(graph, patternToSubstitute.outputDataNodes.begin(),
        patternToSubstitute.outputDataNodes.end(), get_from_pair);

    // FIXME: workaround??
    // erase Island information
    for (auto node : graph.nodes()) {
        graph.metadata(node).erase<Island>();
    }
}

}  // namespace gimpl
}  // namespace cv
