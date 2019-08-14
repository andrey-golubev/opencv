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

#if 0
        ade::NodeHandle newNode = dst.createNode();
        dst.metadata(newNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::OP });

        // std::string island = src.metadata(original).get<cv::gimpl::Island>().name();

        // FIXME: why not just set same op as in src graph???
        dst.metadata(newNode).set(cv::gimpl::Op{ op.k, op.args,{},{} });
        // if (!island.empty()) {
            // dst.metadata(newNode).set(cv::gimpl::Island{ std::move(an iniisland) });
        // }

        return newNode;
#endif
    }
    case NodeType::DATA: {
        const auto& data = src.metadata(original).get<cv::gimpl::Data>();
        return GModel::mkDataNode(dst, data.shape);

#if 0
        ade::NodeHandle newNode = dst.createNode();
        dst.metadata(newNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::DATA });

        const auto shape = data.shape;
        const auto id = dst.metadata().get<cv::gimpl::DataObjectCounter>().GetNewId(shape);
        // FIXME: why new data is not just a copy of old data? due to id?
        dst.metadata(newNode).set(Data{ shape, id, {}, {}, data.storage });

        return newNode;
#endif
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

void customLinkIn(Graph &g, ade::NodeHandle opH, ade::NodeHandle objH, std::size_t in_port)
{
    // Check if input is already connected
#if 0
    for (const auto& in_e : opH->inEdges())
    {
        GAPI_Assert(g.metadata(in_e).get<Input>().port != in_port);
    }
#endif

    auto &op = g.metadata(opH).get<Op>();
    auto &gm = g.metadata(objH).get<Data>();

     // FIXME: check validity using kernel prototype
    GAPI_Assert(in_port < op.args.size());

    ade::EdgeHandle eh = g.link(objH, opH);
    g.metadata(eh).set(Input{in_port});

    // Replace an API object with a REF (G* -> GOBJREF)
    op.args[in_port] = cv::GArg(RcDesc{gm.rc, gm.shape, {}});
}

}  // anonymous namespace

void performSubstitution(Graph& graph, const Graph& substitute, const cv::gimpl::GModel::Graph& pattern,
    const SubgraphMatch& patternToGraph, const SubgraphMatch& patternToSubstitute) {
#if 0
    UNUSED(substitute);
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
#endif

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
#if 0
        // create input node and remember it
        createdCorrespodingNodes.insert({curr, create(substitute, graph, curr)});
#else
        createdCorrespodingNodes.insert(
            {curr, find(curr, patternToSubstitute.inputDataNodes, patternToGraph.inputDataNodes)});
#endif
        // populate nodes with readers of current input node
        auto currReaders = curr->outNodes();
        std::copy(currReaders.begin(), currReaders.end(), std::back_inserter(nodes));
    }

    // we only care about data nodes here: they must be visited once
    SubgraphMatch::S visitedDataNodes;

    // b. traverse the graph starting from the second level of nodes
    while (!nodes.empty()) {
        auto curr = nodes.front();
        nodes.pop_front();

        // FIXME: this should be done via ade::util::filter() - but didn't work
        if (visitedDataNodes.cend() != visitedDataNodes.find(curr)) {
            continue;
        }

        // create new node and remember it
#if 0
        ade::NodeHandle createdCurr = create(substitute, graph, curr);
#else
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

                auto dataCreated = graph.metadata(createdCurr).get<Data>();

                auto patternCurr = findKey(curr, patternToSubstitute.outputDataNodes);
                auto data = pattern.metadata(patternCurr).get<Data>();
                int a = 5;
            } else {
                createdCurr = create(substitute, graph, curr);
            }
        }
#endif
        createdCorrespodingNodes.insert({curr, createdCurr});

        // link new node with the node from the previous level
        auto currInEdges = curr->inEdges();
        for (const auto& edge : currInEdges) {
            auto writer = edge->srcNode();
            const auto& createdWriter = createdCorrespodingNodes.at(writer);

            // create edges
            switch (substitute.metadata(curr).get<NodeType>().t) {
            case NodeType::OP: {
                const auto port = substitute.metadata(edge).get<Input>().port;
                // GModel::linkIn(graph, createdCurr, createdWriter, port);
                customLinkIn(graph, createdCurr, createdWriter, port);
                break;
            }
            case NodeType::DATA: {
                const auto port = substitute.metadata(edge).get<Output>().port;
                const auto& data = substitute.metadata(curr).get<Data>();

                if (existing) {
                    for (auto e : createdCurr->inEdges()) {
                        graph.erase(e);
                    }
                }
                GModel::linkOut(graph, createdWriter, createdCurr, port);
                visitedDataNodes.insert(curr);
                break;
            }
            default: GAPI_Assert(false && "unrecognized NodeType");
            }
        }

        // populate nodes with readers of current node
        auto currReaders = curr->outNodes();
        std::copy(currReaders.begin(), currReaders.end(), std::back_inserter(nodes));
    }

    // FIXME: how old internal nodes are going to be deleted??

#if 0
    // 2) now redirect readers && do clean-up
    // redirect input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);

        do {
            // FIXME: is it alright to just skip?
            // if (graphDataNode->inEdges().size() <= 0) break;  // do not redirect if no writers
            GModel::redirectWriter(
                graph, graphDataNode, createdCorrespodingNodes.at(substituteDataNode));
        } while (false);
        // graph.erase(graphDataNode);
    }

    // redirect output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);

        do {
            // if (graphDataNode->outEdges().size() <= 0) break;  // do not redirect if no readers
            GModel::redirectReaders(
                graph, graphDataNode, createdCorrespodingNodes.at(substituteDataNode));
        } while (false);
        // graph.erase(graphDataNode);
    }
#endif

    auto size = createdCorrespodingNodes.size();
    std::cout << "Created nodes size = " << size << std::endl;

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

}  // namespace gimpl
}  // namespace cv