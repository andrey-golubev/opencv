// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#include <list>

#define UNUSED(x) (void)x

namespace cv { namespace gimpl {

namespace {
using Graph = GModel::Graph;

ade::NodeHandle create(const Graph& src, Graph& dst, const ade::NodeHandle& original) {
    switch (src.metadata(original).get<NodeType>().t) {
    case NodeType::OP: {
        ade::NodeHandle newNode = dst.createNode();
        dst.metadata(newNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::OP });

        auto op = src.metadata(original).get<cv::gimpl::Op>();
        auto island = src.metadata(original).get<cv::gimpl::Island>();

#if 0  // FIXME: why not just set same op as in src graph???
        dst.metadata(newNode).set(cv::gimpl::Op{ op.k, op.args,{},{} });
#endif

#if 0
        for (auto& out : op.outs) {
            // FIXME: what's a proper way to update obj counter? do we need to do it at all?
            out.id = dst.metadata().get<cv::gimpl::DataObjectCounter>().GetCurrId(out.shape);
        }
#endif
        dst.metadata(newNode).set(cv::gimpl::Op{ op });
        dst.metadata(newNode).set(island);

        return newNode;
    }
    case NodeType::DATA: {
        ade::NodeHandle newNode = dst.createNode();
        dst.metadata(newNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::DATA });

        const auto& data = src.metadata(original).get<cv::gimpl::Data>();
        const auto shape = data.shape;
        const auto id = dst.metadata().get<cv::gimpl::DataObjectCounter>().GetNewId(shape);
        // FIXME: why new data is not just a copy of old data? due to id?
        dst.metadata(newNode).set(
            cv::gimpl::Data{ shape, id, data.meta, data.ctor, data.storage });

        return newNode;
    }
    default: GAPI_Assert(false && "unrecognized NodeType");
    }
}

}  // anonymous namespace

void performSubstitution(Graph& graph, const Graph& substitute, const SubgraphMatch& patternToGraph,
    const SubgraphMatch& patternToSubstitute) {
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

    // Idea: 1) construct substitute graph in main graph; 2) redirect readers/writers from graph
    // nodes to corresponding newly constructed pseudo-substitute nodes

    // 1) traverse the graph from the nodes after the inputs (second level)
    std::list<ade::NodeHandle> nodes;
    const auto& substituteInputs = substitute.metadata().get<Protocol>().in_nhs;

    // remember substitute node for each main graph node
    SubgraphMatch::M createdCorrespodingNodes;

    // a. create input nodes
    for (const auto& curr : substituteInputs) {
        // create input node and remember it
        createdCorrespodingNodes.insert({curr, create(substitute, graph, curr)});

        // populate nodes with readers of current input node
        auto currReaders = curr->outNodes();
        std::copy(currReaders.begin(), currReaders.end(), std::back_inserter(nodes));
    }

    // b. traverse the graph starting from the second level of nodes
    while (!nodes.empty()) {
        auto curr = nodes.front();
        nodes.pop_front();

        // create new node and remember it
        auto createdCurr = create(substitute, graph, curr);
        createdCorrespodingNodes.insert({curr, createdCurr});

        // link new node with the node from the previous level
        auto currInEdges = curr->inEdges();
        for (const auto& edge : currInEdges) {
            auto writer = edge->srcNode();
            const auto& createdWriter = createdCorrespodingNodes.at(writer);
            // create edge
            auto createdEdge = graph.link(createdWriter, createdCurr);
            // set ports
            graph.metadata(createdEdge).set(
                Input{ substitute.metadata(edge).get<Input>().port });
            graph.metadata(createdEdge).set(
                Output{ substitute.metadata(edge).get<Output>().port });

            // FIXME: now do something really weird  -- setup descriptors in Op nodes
            switch (substitute.metadata(curr).get<NodeType>().t) {
            case NodeType::OP: {
                const auto port = graph.metadata(createdEdge).get<Input>().port;
                auto& createdCurrOp = graph.metadata(createdCurr).get<cv::gimpl::Op>();
                auto& createdWriterData = graph.metadata(createdWriter).get<cv::gimpl::Data>();

                // FIXME: something similar is done in create() -- revise usage
                createdCurrOp.args[port] = cv::GArg(
                    cv::gimpl::RcDesc{ createdWriterData.rc, createdWriterData.shape,
                        createdWriterData.ctor });
                break;
            }
            case NodeType::DATA: {
                const auto port = graph.metadata(createdEdge).get<Output>().port;
                auto& createdWriterOp = graph.metadata(createdWriter).get<cv::gimpl::Op>();
                auto& createdCurrData = graph.metadata(createdCurr).get<cv::gimpl::Data>();

                // FIXME: why this part is needed at all, let's just set outs from original node...
                const auto storageWithPort = ade::util::checked_cast<std::size_t>(port + 1);
                const auto minOutSize = std::max(createdWriterOp.outs.size(), storageWithPort);
                createdWriterOp.outs.resize(minOutSize, cv::gimpl::RcDesc{-1, GShape::GMAT, {}});
                // FIXME: something similar is done in create() -- revise usage
                createdWriterOp.outs[port] =
                    cv::gimpl::RcDesc{createdCurrData.rc, createdCurrData.shape,
                        createdCurrData.ctor};
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

    // 2) now redirect readers && do clean-up
    // redirect input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);

        GModel::redirectWriter(
            graph, graphDataNode, createdCorrespodingNodes.at(substituteDataNode));
        graph.erase(graphDataNode);
    }

    // redirect output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);

        GModel::redirectReaders(
            graph, graphDataNode, createdCorrespodingNodes.at(substituteDataNode));
        graph.erase(graphDataNode);
    }
}

}  // namespace gimpl
}  // namespace cv