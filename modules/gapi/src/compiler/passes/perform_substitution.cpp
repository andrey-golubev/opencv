// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#define UNUSED(x) (void)x

namespace cv { namespace gimpl {

namespace {
using Graph = GModel::Graph;
}  // anonymous namespace

void performSubstitution(Graph& graph, const SubgraphMatch& patternToGraph,
    const SubgraphMatch& patternToSubstitute) {
    // substitute input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);

        auto it = std::find(graph.nodes().begin(), graph.nodes().end(), graphDataNode);
        GAPI_Assert(it != graph.nodes().end());
        *it = substituteDataNode;
    }

    // FIXME: internal layers must be matched automatically? (due to inputs/output are loosely
    //        coupled to internal nodes?)

    // substitute output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);

        auto it = std::find(graph.nodes().begin(), graph.nodes().end(), graphDataNode);
        GAPI_Assert(it != graph.nodes().end());
        *it = substituteDataNode;
    }
}

}  // namespace gimpl
}  // namespace cv