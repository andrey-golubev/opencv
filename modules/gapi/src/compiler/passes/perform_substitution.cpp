// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#include <list>
#include <algorithm>

#define UNUSED(x) (void)x

namespace cv { namespace gimpl {

namespace {
using Graph = GModel::Graph;

// FIXME: use GModel::mk*Node instead
// Creates new node based on original metadata in dst graph
ade::NodeHandle create(const Graph& src, Graph& dst, const ade::NodeHandle& original) {
    switch (src.metadata(original).get<NodeType>().t) {
    case NodeType::OP: {
        auto op = src.metadata(original).get<cv::gimpl::Op>();
        return GModel::mkOpNode(dst, op.k, op.args, "");
    }
    case NodeType::DATA: {
        const auto& data = src.metadata(original).get<cv::gimpl::Data>();
        return GModel::mkDataNode(dst, data.shape);
    }
    default: GAPI_Assert(false && "unrecognized NodeType");
    }
}

// Finds key of src_node in src_map (searching by value)
ade::NodeHandle findKey(const ade::NodeHandle& src_node, const SubgraphMatch::M& src_map) {
    using NH = ade::NodeHandle;
    auto it = std::find_if(src_map.cbegin(), src_map.cend(), [&] (const std::pair<NH, NH>& p) {
        return p.second == src_node;
    });
    GAPI_Assert(it != src_map.cend());
    return it->first;
}

// Finds src_node in src_map (searching by value) then returns dst_map[found]
ade::NodeHandle find(const ade::NodeHandle& src_node, const SubgraphMatch::M& src_map,
    const SubgraphMatch::M& dst_map) {
    using NH = ade::NodeHandle;
    auto key = findKey(src_node, src_map);
    return dst_map.at(key);
}

template<typename It>
void erase_many_nodes(Graph& g, It first, It last) {
    for (; first != last; ++first) {
        g.erase(*first);
    }
}

template<typename It>
void erase_many_pairs(Graph& g, It first, It last) {
    for (; first != last; ++first) {
        g.erase(first->second);
    }
}
}  // anonymous namespace

void performSubstitution(Graph& graph, const Graph& substitute, const cv::gimpl::GModel::Graph& pattern,
    const SubgraphMatch& patternToGraph, const SubgraphMatch& patternToSubstitute) {
    // FIXME: start from "input" OP nodes, not DATA -> this should look prettier??
    // Idea: 1) construct substitute graph in main graph; 2) redirect readers/writers from graph
    // nodes to corresponding newly constructed pseudo-substitute nodes

    // 1) traverse the graph from the nodes after the inputs (second level)
    std::list<ade::NodeHandle> nodes;
    const auto& substituteInputs = substitute.metadata().get<Protocol>().in_nhs;
    const auto& substituteOutputs = substitute.metadata().get<Protocol>().out_nhs;

    // remember substitute node for each main graph node
    SubgraphMatch::M createdCorrespodingNodes;

    // a. _do not_ create input DATA nodes, just find corresponding graph node
    for (const auto& curr : substituteInputs) {
        createdCorrespodingNodes.insert(
            {curr, find(curr, patternToSubstitute.inputDataNodes, patternToGraph.inputDataNodes)});
        // populate nodes with readers of current input node
        auto currReaders = curr->outNodes();
        std::copy(currReaders.begin(), currReaders.end(), std::back_inserter(nodes));
    }

    // we only care about data nodes here: they must be visited once
    SubgraphMatch::S visited;

    // b. traverse the graph starting from the second level of nodes
    while (!nodes.empty()) {
        auto curr = nodes.front();
        nodes.pop_front();

        // FIXME: this should be done via ade::util::filter() - but didn't work
        if (visited.cend() != visited.find(curr)) {
            continue;
        }

        // create new node and remember it
        bool existing = false;
        ade::NodeHandle createdCurr;
        if (createdCorrespodingNodes.count(curr) > 0) {
            createdCurr = createdCorrespodingNodes.at(curr);
        } else {
            // if curr node is an output DATA node, do not create it, just find corresponding graph node
            if (substituteOutputs.cend() !=
                std::find(substituteOutputs.cbegin(), substituteOutputs.cend(), curr)) {
                createdCurr =
                    find(curr, patternToSubstitute.outputDataNodes, patternToGraph.outputDataNodes);
                existing = true;
            } else {
                createdCurr = create(substitute, graph, curr);
            }
        }
        createdCorrespodingNodes.insert({curr, createdCurr});

        // link new node with the node from the previous level
        auto currInEdges = curr->inEdges();
        for (const auto& edge : currInEdges) {
            auto writer = edge->srcNode();
            const auto& createdWriter = createdCorrespodingNodes.at(writer);

            // create edges
            switch (substitute.metadata(curr).get<NodeType>().t) {
            case NodeType::OP: {
                GModel::linkIn(graph, createdCurr, createdWriter,
                    substitute.metadata(edge).get<Input>().port);
                break;
            }
            case NodeType::DATA: {
                if (existing) {
                    for (auto e : createdCurr->inEdges()) {
                        graph.erase(e);
                    }
                }
                GModel::linkOut(graph, createdWriter, createdCurr,
                    substitute.metadata(edge).get<Output>().port);
                break;
            }
            default: GAPI_Assert(false && "unrecognized NodeType");
            }
        }
        visited.insert(curr);

        // populate nodes with readers of current node
        auto currReaders = curr->outNodes();
        std::copy(currReaders.begin(), currReaders.end(), std::back_inserter(nodes));
    }

    // 3) erase internal nodes
    erase_many_pairs(graph, patternToGraph.startOpNodes.begin(), patternToGraph.startOpNodes.end());
    erase_many_nodes(graph, patternToGraph.internalLayers.begin(),
        patternToGraph.internalLayers.begin());
    erase_many_pairs(graph, patternToGraph.finishOpNodes.begin(),
        patternToGraph.finishOpNodes.begin());

    // FIXME: workaround??
    for (auto node : graph.nodes()) {
        graph.metadata(node).erase<Island>();
    }
}

void performSubstitutionAlt(Graph& graph,
    const SubgraphMatch& patternToGraph, const SubgraphMatch& patternToSubstitute) {
    // substitute input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);
        GModel::redirectReaders(graph, substituteDataNode, graphDataNode);
    }

    // substitute output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);
        for (auto e : graphDataNode->inEdges()) {
            graph.erase(e);
        }
        GModel::redirectWriter(graph, substituteDataNode, graphDataNode);
    }

    // erase internal nodes
    erase_many_pairs(graph, patternToSubstitute.inputDataNodes.begin(), patternToSubstitute.inputDataNodes.end());
    erase_many_pairs(graph, patternToGraph.startOpNodes.begin(), patternToGraph.startOpNodes.end());
    erase_many_nodes(graph, patternToGraph.internalLayers.begin(),
        patternToGraph.internalLayers.begin());
    erase_many_pairs(graph, patternToGraph.finishOpNodes.begin(),
        patternToGraph.finishOpNodes.begin());
    erase_many_pairs(graph, patternToSubstitute.outputDataNodes.begin(), patternToSubstitute.outputDataNodes.end());

    // FIXME: workaround??
    for (auto node : graph.nodes()) {
        graph.metadata(node).erase<Island>();
    }
}

}  // namespace gimpl
}  // namespace cv