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
using Metadata = typename Graph::CMetadataT;
using VisitedMatchings = std::list<std::pair<ade::NodeHandle, ade::NodeHandle>>;

// Returns true if two DATA nodes are semantically and structurally identical:
//  - both nodes have the same GShape
//  - both nodes have the same storage
//  - both nodes have the same meta
//
// @param first - first node to compare
// @param firstMeta - metadata of first
// @param second - second node to compare
// @param secondMeta - metadata of second
bool compareDataNodes(const ade::NodeHandle& first,
                      const Metadata& firstMeta,
                      const ade::NodeHandle& second,
                      const Metadata& secondMeta) {
    UNUSED(first);
    UNUSED(second);

    GAPI_Assert(firstMeta.get<NodeType>().t == NodeType::DATA);
    GAPI_Assert(firstMeta.get<NodeType>().t == secondMeta.get<NodeType>().t);

    const auto& firstData = firstMeta.get<Data>();
    const auto& secondData = secondMeta.get<Data>();
    // compare shape
    if (firstData.shape != secondData.shape) {
        return false;
    }
    // compare storage
    if (firstData.storage != secondData.storage) {
        return false;
    }
    // compare meta
    // FIXME: GArrayDesc && GScalarDesc always return true in operator==
    if (firstData.meta != secondData.meta) {
        return false;
    }


#if 0
    if (firstPorts.begin() != secondPorts.begin()) {
        return false;
    }

    const auto& firstOutputEdges = first->outEdges();
    const auto& secondOutputEdges = second->outEdges();

    if (firstOutputEdges.size() != secondOutputEdges.size()) {
        return false;
    }
#endif

    return true;
};

SubgraphMatch::M matchDataNodes(const Graph& pattern, const Graph& substitute,
    const std::vector<ade::NodeHandle>& patternNodes,
    std::vector<ade::NodeHandle> substituteNodes) {
    SubgraphMatch::M matched;
    // FIXME: something smarter?
    auto size = substituteNodes.size();  // must be the same as patternNodes.size() at this point
    for (const auto& patternNode : patternNodes) {
        const auto& patternMeta = pattern.metadata(patternNode);
        // look at first size elements, found nodes are pushed to the end
        auto it = std::find_if(substituteNodes.begin(), substituteNodes.begin() + size,
            [&] (const ade::NodeHandle& substituteNode) {
                const auto& substituteMeta = substitute.metadata(substituteNode);
                return compareDataNodes(patternNode, patternMeta, substituteNode, substituteMeta);
            });
        if (it == substituteNodes.end()) {
            return {};  // nothing found for some node <=> nothing found at all
        }
        matched.insert({ patternNode, *it });

        // search optimization: push found nodes to the end
        // FIXME: same iterator supported?
        std::iter_swap(it, substituteNodes.begin() + (size - 1));
        size--;
    }
    return matched;
}

}  // anonymous namespace

SubgraphMatch findPatternToSubstituteMatch(const Graph& pattern, const Graph& substitute) {
    //---------------------------------------------------------------
    // Match data nodes which start and end our pattern and substitute
    const auto& patternDataInputs = pattern.metadata().get<Protocol>().in_nhs;
    const auto& patternDataOutputs = pattern.metadata().get<Protocol>().out_nhs;

    const auto& substituteDataInputs = substitute.metadata().get<Protocol>().in_nhs;
    const auto& substituteDataOutputs = substitute.metadata().get<Protocol>().out_nhs;

    // if number of data nodes doesn't match, abort
    if (patternDataInputs.size() != substituteDataInputs.size()
        || patternDataOutputs.size() != substituteDataOutputs.size()) {
        return {};
    }

    // for each pattern input we must find a corresponding substitute input
    auto matchedDataInputs = matchDataNodes(pattern, substitute, patternDataInputs,
        substituteDataInputs);
    // if nothing found, abort
    if (matchedDataInputs.empty()) {
        return {};
    }
    auto matchedDataOutputs = matchDataNodes(pattern, substitute, patternDataOutputs,
        substituteDataOutputs);
    // if nothing found, abort
    if (matchedDataOutputs.empty()) {
        return {};
    }

    //---------------------------------------------------------------
    // Construct SubgraphMatch object
    SubgraphMatch match;
    match.inputDataNodes = std::move(matchedDataInputs);
    match.outputDataNodes = std::move(matchedDataOutputs);

    match.inputTestDataNodes = std::move(substituteDataInputs);
    match.outputTestDataNodes = std::move(substituteDataOutputs);

    // FIXME: populate these nodes
    auto& startOps = match.startOpNodes;
    auto& endOps = match.finishOpNodes;
    auto& internalNodes = match.internalLayers;  // NB: these should also be placed layer by layer!!

    UNUSED(startOps);
    UNUSED(endOps);
    UNUSED(internalNodes);

    return match;
}

SubgraphMatch findPatternToSubstituteMatch(const cv::gimpl::GModel::Graph& pattern,
    const cv::gimpl::GModel::Graph& substitute,
    const std::vector<ade::NodeHandle>& substitute_ins,
    const std::vector<ade::NodeHandle>& substitute_outs) {
    //---------------------------------------------------------------
    // Match data nodes which start and end our pattern and substitute
    const auto& patternDataInputs = pattern.metadata().get<Protocol>().in_nhs;
    const auto& patternDataOutputs = pattern.metadata().get<Protocol>().out_nhs;

    const auto& substituteDataInputs = substitute_ins;
    const auto& substituteDataOutputs = substitute_outs;

    // if number of data nodes doesn't match, abort
    if (patternDataInputs.size() != substituteDataInputs.size()
        || patternDataOutputs.size() != substituteDataOutputs.size()) {
        return {};
    }

    // for each pattern input we must find a corresponding substitute input
    auto matchedDataInputs = matchDataNodes(pattern, substitute, patternDataInputs,
        substituteDataInputs);
    // if nothing found, abort
    if (matchedDataInputs.empty()) {
        return {};
    }
    auto matchedDataOutputs = matchDataNodes(pattern, substitute, patternDataOutputs,
        substituteDataOutputs);
    // if nothing found, abort
    if (matchedDataOutputs.empty()) {
        return {};
    }

    //---------------------------------------------------------------
    // Construct SubgraphMatch object
    SubgraphMatch match;
    match.inputDataNodes = std::move(matchedDataInputs);
    match.outputDataNodes = std::move(matchedDataOutputs);

    match.inputTestDataNodes = std::move(substituteDataInputs);
    match.outputTestDataNodes = std::move(substituteDataOutputs);

    // FIXME: populate these nodes
    auto& startOps = match.startOpNodes;
    auto& endOps = match.finishOpNodes;
    auto& internalNodes = match.internalLayers;  // NB: these should also be placed layer by layer!!

    UNUSED(startOps);
    UNUSED(endOps);
    UNUSED(internalNodes);

    return match;
}

}  // namespace gimpl
}  // namespace cv
