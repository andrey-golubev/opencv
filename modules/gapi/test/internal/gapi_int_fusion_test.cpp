// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <stdexcept>
#include <cmath>

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "api/gcomputation_priv.hpp"
#include "compiler/gcompiled_priv.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/passes/passes.hpp"

#include "compiler/passes/pattern_matching.hpp"

#include "../common/gapi_tests_common.hpp"

#include "logger.hpp"

namespace opencv_test
{
 //Pattern for Fluid:                                                             Substitution for Fluid:
//          +----------------------------------------------+                               +-----------------+
//          |                                              |                               |                 |
//+------+  |    XXXXXXXXXX      +------+     XXXXXXXXXX   |  +------+           +------+  |    XXXXXXXXXX   |    +------+
//|      |  |   X          X     |      |    X          X  |  |      |           |      |  |   X          X  |    |      |
//| GMat +----->X  Resize  X+--->+ GMat +--->X  toNCHW  X+--->+ GMat |           | GMat +----->XResize3c3pX+----->+ GMat |
//+------+  |    XXXXXXXXXX      +------+     XXXXXXXXXX   |  +------+           +------+  |    XXXXXXXXXX   |    +------+
//          |                                              |                               |                 |
//          +----------------------------------------------+                               +-----------------+


 //Test:
//+------+
//|      |  +------------------------------------------------------------------------+
//| GMat |  |                                                                        |
//+------+  |   XXXXXXXXXX     +------+    XXXXXXXXXX      +------+     XXXXXXXXXX   |  +------+
//       +---->X          X    |      |   X          X     |      |    X          X  |  |      |
//       +---->XNV12tToBGR+--->+ GMat +---X  Resize  X+--->+ GMat +--->X  toNCHW  X+--->+ GMat |
//+------+  |   XXXXXXXXXX     +------+    XXXXXXXXXX      +------+     XXXXXXXXXX   |  +------+
//|      |  |                                                                        |
//| GMat |  +------------------------------------------------------------------------+
//+------+


//Result:
//+------+
//|      |  +--------------------------------------------+
//| GMat |  |                                            |
//+------+  |   XXXXXXXXXX     +------+     XXXXXXXXXX   |  +------+
//       +---->X          X    |      |    X          X  |  |      |
//       +---->XNV12tToBGR+--->+ GMat +--->XResize3c3pX---->+ GMat |
//+------+  |   XXXXXXXXXX     +------+     XXXXXXXXXX   |  +------+
//|      |  |                                            |
//| GMat |  +--------------------------------------------+
//+------+

//Switch to use of this function
//static void toPlanar(const cv::Mat& in, cv::Mat& out)
//{
//    std::vector<cv::Mat> outs(3);
//    for (int i = 0; i < 3; i++) {
//        outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
//    }
//    cv::split(in, outs);
//}

G_TYPED_KERNEL(GToNCHW, <GMatP(GMat)>, "test.toNCHW") {
    static GMatDesc outMeta(GMatDesc in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar == false);
        return in.asPlanar();
    }
};

static GMat toNCHW(const GMat& src)
{
    return GToNCHW::on(src);
}

GAPI_OCV_KERNEL(OCVToNCHW, GToNCHW)
{
    static void run(const cv::Mat& in, cv::Mat& out)
    {
        GAPI_Assert(out.depth() == in.depth());
        GAPI_Assert(out.channels() == 1);
        GAPI_Assert(in.channels() == 3);
        GAPI_Assert(out.cols == in.cols);
        GAPI_Assert(out.rows == 3 * in.rows);

        std::vector<cv::Mat> outs(3);
        for (int i = 0; i < 3; i++) {
            outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
        }
        cv::split(in, outs);
    }
};

G_TYPED_KERNEL(GResize3c3p, <GMatP(GMat, Size, int)>, "test.resize3c3p") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar == false);
        return in.withSize(sz).asPlanar();
    }
};

static GMat resize3c3p(const GMat& src, cv::Size size, int interp = 1)
{
    return GResize3c3p::on(src, size, interp);
}

GAPI_OCV_KERNEL(OCVResize3c3p, GResize3c3p)
{
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat& out)
    {
        cv::Mat resized_mat;
        cv::resize(in, resized_mat, out_sz, 0, 0, interp);

        std::vector<cv::Mat> outs(3);
        for (int i = 0; i < 3; i++) {
            outs[i] = out(cv::Rect(0, i*out_sz.height, out_sz.width, out_sz.height));
        }
        cv::split(resized_mat, outs);
    }
};

#if 0
static void retrieveUttermostOpNodes(cv::gimpl::GModel::Graph graph,
                             std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>& firstOpNodes,
                             std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>& lastOpNodes) {

    auto firstDataNodes = graph.metadata().get<cv::gimpl::Protocol>().in_nhs;
    auto lastDataNodes = graph.metadata().get<cv::gimpl::Protocol>().out_nhs;

    for (auto node : firstDataNodes) {
        auto opNodes = node->outNodes();
        firstOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    for (auto node : lastDataNodes) {
        auto opNodes = node->inNodes();
        lastOpNodes.insert(opNodes.begin(), opNodes.end());
    }
}

static void substituteMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph, cv::gimpl::SubgraphMatch subgraphMatch, cv::gimpl::GModel::Graph substGraph, cv::gapi::SubgraphMatch substituteMatch) {
    auto compInputApiMatches = subgraphMatch.inputDataNodesMatches;
    auto substInputApiMatches = substituteMatch.inputDataNodesMatches;
    auto compFirstOpNodesMatches = subgraphMatch.firstOpNodesMatches;
    auto substFirstOpNodesMatches = substituteMatch.firstOpNodesMatches;

    auto compInternals = subgraphMatch.internalLayers;
    for (auto it = compInternals.begin(); it != compInternals.end(); ++it) {
        auto node = *it;
        compGraph.erase(node);
    }

    auto createTwinOpNode = [](ade::NodeHandle originOpNode, cv::gimpl::GModel::Graph fromGraph, cv::gimpl::GModel::Graph dstGraph) -> ade::NodeHandle {
        ade::NodeHandle twinOpNode = dstGraph.createNode();
        dstGraph.metadata(twinOpNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::OP });

        auto op = fromGraph.metadata(originOpNode).get<cv::gimpl::Op>();
        auto island = fromGraph.metadata(originOpNode).get<cv::gimpl::Island>();

        dstGraph.metadata(twinOpNode).set(cv::gimpl::Op{ op.k, op.args,{},{} });
        dstGraph.metadata(twinOpNode).set(island);

        return twinOpNode;
    };

    auto createTwinDataNode = [](ade::NodeHandle originDataNode, cv::gimpl::GModel::Graph fromGraph, cv::gimpl::GModel::Graph dstGraph) -> ade::NodeHandle {
        ade::NodeHandle twinDataNode = dstGraph.createNode();
        dstGraph.metadata(twinDataNode).set(cv::gimpl::NodeType{ cv::gimpl::NodeType::DATA });

        auto shape = fromGraph.metadata(originDataNode).get<cv::gimpl::Data>().shape;
        const auto id = dstGraph.metadata().get<cv::gimpl::DataObjectCounter>().GetNewId(shape);
        GMetaArg meta;
        cv::gimpl::HostCtor ctor;
        cv::gimpl::Data::Storage storage = cv::gimpl::Data::Storage::INTERNAL;

        dstGraph.metadata(twinDataNode).set(cv::gimpl::Data{ shape, id, meta, ctor, storage });

        return twinDataNode;
    };

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> substitutedNodesMatchings;

    // TODO: Support case below:
    // Two pattern Op nodes with multiple edges connected from two pattern Data nodes.(012 345 - pattern, 012 345 - data, 345 012 - subst)
    // Shall support edges mapping structure this case but may be implemented without it.
    for (auto matchIt = substFirstOpNodesMatches.begin(); matchIt != substFirstOpNodesMatches.end(); ++matchIt) {

        auto compOpNode = compFirstOpNodesMatches[matchIt->first];
        auto substOpNode = matchIt->second;

        auto newCompOpNode = createTwinOpNode(substOpNode, substGraph, compGraph);

        auto patternInNodes = cv::gimpl::GModel::orderedInputs(patternGraph, matchIt->first);
        for (auto patternInNodeIt = patternInNodes.begin(); patternInNodeIt != patternInNodes.end(); ++patternInNodeIt) {
            auto patternInNode = *patternInNodeIt;
            auto compInNode = compInputApiMatches[patternInNode];

            auto compInEdge = *std::find_if(compInNode->outEdges().begin(), compInNode->outEdges().end(), [&compOpNode](ade::EdgeHandle edge) {
                                                                                                            return edge->dstNode() == compOpNode;
                                                                                                          });
            compGraph.erase(compInEdge);

            auto substInNode = substInputApiMatches[patternInNode];
            auto substInEdge = *std::find_if(substInNode->outEdges().begin(), substInNode->outEdges().end(), [&substOpNode](ade::EdgeHandle edge) {
                                                                                                                return edge->dstNode() == substOpNode;
                                                                                                            });
            auto inPort = substGraph.metadata(substInEdge).get<cv::gimpl::Input>().port;


            auto newCompEdge = compGraph.link(compInNode, newCompOpNode);
            compGraph.metadata(newCompEdge).set(cv::gimpl::Input{ inPort });

            auto &newCompOp = compGraph.metadata(newCompOpNode).get<cv::gimpl::Op>();
            auto &compInData = compGraph.metadata(compInNode).get<cv::gimpl::Data>();

            newCompOp.args[inPort] = cv::GArg(cv::gimpl::RcDesc{ compInData.rc, compInData.shape,{} });
        }

        compGraph.erase(compOpNode);

        substitutedNodesMatchings[substOpNode] = newCompOpNode;
    }

    auto substituteInternals = substituteMatch.internalLayers;
    auto substLastOpNodesMatches = substituteMatch.lastOpNodesMatches;

    for (auto it = substLastOpNodesMatches.begin(); it != substLastOpNodesMatches.end(); ++it) {
        if (std::find_if(substFirstOpNodesMatches.begin(), substFirstOpNodesMatches.end(), [&it](std::pair<ade::NodeHandle, ade::NodeHandle> firstOpNodeMatch)
                                                                                            {return firstOpNodeMatch.second == it->second; })
            == substFirstOpNodesMatches.end()) {
            substituteInternals.push_back(it->second);
        }
    }
    for (auto it = substituteInternals.begin(); it != substituteInternals.end(); ++it) {
        auto substNode = *it;

        ade::NodeHandle newCompNode;
        if (substGraph.metadata(substNode).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
            newCompNode = createTwinDataNode(substNode, substGraph, compGraph);
        }
        else {
            newCompNode = createTwinOpNode(substNode, substGraph, compGraph);
        }

        auto substInEdges = substNode->inEdges();
        for (auto substInEdgeIt = substInEdges.begin(); substInEdgeIt != substInEdges.end(); ++substInEdgeIt) {
            auto substInEdge = *substInEdgeIt;
            auto substInNode = substInEdge->srcNode();
            auto compInNode = substitutedNodesMatchings[substInNode];

            auto newCompEdge = compGraph.link(compInNode, newCompNode);

            std::size_t port;

            if (substGraph.metadata(substNode).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
                port = substGraph.metadata(substInEdge).get<cv::gimpl::Output>().port;
                compGraph.metadata(newCompEdge).set(cv::gimpl::Output{ port });

                auto& newCompOp = compGraph.metadata(compInNode).get<cv::gimpl::Op>();
                auto& outData = compGraph.metadata(newCompNode).get<cv::gimpl::Data>();

                const auto storageWithPort = ade::util::checked_cast<std::size_t>(port + 1);
                const auto minOutSize = std::max(newCompOp.outs.size(), storageWithPort);
                newCompOp.outs.resize(minOutSize, cv::gimpl::RcDesc{ -1, GShape::GMAT,{} }); // FIXME: Invalid shape instead?
                newCompOp.outs[port] = cv::gimpl::RcDesc{ outData.rc, outData.shape,{} };
            }
            else {
                port = substGraph.metadata(substInEdge).get<cv::gimpl::Input>().port;
                compGraph.metadata(newCompEdge).set(cv::gimpl::Input{ port });

                auto &newCompOp = compGraph.metadata(newCompNode).get<cv::gimpl::Op>();
                auto &compInData = compGraph.metadata(compInNode).get<cv::gimpl::Data>();

                newCompOp.args[port] = cv::GArg(cv::gimpl::RcDesc{ compInData.rc, compInData.shape,{} });
            }
        }

        substitutedNodesMatchings[substNode] = newCompNode;
    }

    auto compOutputApiMatches = subgraphMatch.outputDataNodesMatches;
    auto substOutputApiMatches = substituteMatch.outputDataNodesMatches;
    auto compLastOpNodesMatches = subgraphMatch.lastOpNodesMatches;

    for (auto matchIt = substLastOpNodesMatches.begin(); matchIt != substLastOpNodesMatches.end(); ++matchIt) {
        auto compOpNode = compLastOpNodesMatches[matchIt->first];
        auto substOpNode = matchIt->second;
        auto newCompOpNode = substitutedNodesMatchings[substOpNode];

        auto patternOutNodes = cv::gimpl::GModel::orderedOutputs(patternGraph, matchIt->first);
        for (auto patternOutNodeIt = patternOutNodes.begin(); patternOutNodeIt != patternOutNodes.end(); ++patternOutNodeIt) {
            auto patternOutNode = *patternOutNodeIt;
            auto compOutNode = compOutputApiMatches[patternOutNode];

            auto compOutEdge = *std::find_if(compOutNode->inEdges().begin(), compOutNode->inEdges().end(), [&compOpNode](ade::EdgeHandle edge) {
                return edge->srcNode() == compOpNode;
            });
            compGraph.erase(compOutEdge);

            auto substOutNode = substOutputApiMatches[patternOutNode];
            auto substOutEdge = *std::find_if(substOutNode->inEdges().begin(), substOutNode->inEdges().end(), [&substOpNode](ade::EdgeHandle edge) {
                return edge->srcNode() == substOpNode;
            });
            auto outPort = substGraph.metadata(substOutEdge).get<cv::gimpl::Output>().port;


            auto newCompEdge = compGraph.link(newCompOpNode, compOutNode);
            compGraph.metadata(newCompEdge).set(cv::gimpl::Output{ outPort });

            auto& newCompOp = compGraph.metadata(newCompOpNode).get<cv::gimpl::Op>();
            auto& outData = compGraph.metadata(compOutNode).get<cv::gimpl::Data>();

            const auto storageWithPort = ade::util::checked_cast<std::size_t>(outPort + 1);
            const auto minOutSize = std::max(newCompOp.outs.size(), storageWithPort);
            newCompOp.outs.resize(minOutSize, cv::gimpl::RcDesc{ -1, GShape::GMAT,{} });
            newCompOp.outs[outPort] = cv::gimpl::RcDesc{ outData.rc, outData.shape,{} };
        }

        compGraph.erase(compOpNode);
    }

    // Data nodes indices will be different, but not from the whole range.
}

TEST(GraphFusion, PreprocPipeline1Fusion)
{
    auto pkg = cv::gapi::kernels<OCVToNCHW>();

    cv::Size dstSize{ 224, 224 };

    //----------------------------Pattern graph---------------------------
    cv::Mat patternIm(1080, 1920, CV_8UC3);
    cv::Mat patternPlanarIm(1080 * 3, 1920, CV_8UC1);
    cv::randu(patternIm, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::GMat patternIn;
    auto patternResized = cv::gapi::resize(patternIn, dstSize);
    auto patternSplit = toNCHW(patternResized);

    cv::GComputation pattern(patternIn, patternSplit);
    pattern.apply(cv::gin(patternIm), cv::gout(patternPlanarIm), cv::compile_args(pkg));
    //-------------------------------------------------------------------


    //-------------------------Input GComputation graph------------------
    cv::Mat testY(1080, 1920, CV_8UC1);
    cv::Mat testUV(540, 960, CV_8UC2);
    cv::Mat testPlanarIm(1080 * 3, 1920, CV_8UC1);

    cv::randu(testY, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(testUV, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::GMat y, uv;
    auto converted = cv::gapi::NV12toBGR(y, uv);
    auto resized = cv::gapi::resize(converted, dstSize);
    auto split = toNCHW(resized);

    // TO FIX: unique_ptr is a temporary hack to allow one resource to be shared between different GCompiler-s
    std::unique_ptr<cv::GComputation> computation(new cv::GComputation(cv::GIn(y, uv), cv::GOut(split)));
    computation->apply(cv::gin(testY, testUV), cv::gout(testPlanarIm), cv::compile_args(pkg));
    //--------------------------------------------------------------------


    //-----------------------Pattern Matching-----------------------------
    auto patternGraph = pattern.priv().m_lastCompiled.priv().model();
    auto compGraph = computation->priv().m_lastCompiled.priv().model();
    cv::gimpl::SubgraphMatch match = cv::gapi::findMatches(patternGraph, compGraph);
    //--------------------------------------------------------------------


    //----------------------Substitution graph----------------------------
    auto substitutePkg = cv::gapi::kernels<OCVResize3c3p>();

    cv::Mat substituteIm(1080, 1920, CV_8UC3);
    cv::Mat substitutePlanarIm(1080 * 3, 1920, CV_8UC1);
    cv::randu(substituteIm, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::GMat substituteIn;
    auto resizedPlanar = resize3c3p(substituteIn, dstSize);
    cv::GComputation substitution(cv::GIn(substituteIn), cv::GOut(resizedPlanar));
    substitution.apply(cv::gin(substituteIm), cv::gout(substitutePlanarIm), cv::compile_args(substitutePkg));
    //----------------------------------------------------------------------


    //-------------------------Substitution---------------------------------
    auto substituteGraph = substitution.priv().m_lastCompiled.priv().model();
    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> substitutionFirstOpNodes, substitutionLastOpNodes;
    retrieveUttermostOpNodes(substituteGraph, substitutionFirstOpNodes, substitutionLastOpNodes);

    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> patternFirstOpNodes, patternLastOpNodes;
    retrieveUttermostOpNodes(patternGraph, patternFirstOpNodes, patternLastOpNodes);

    assert(substitutionFirstOpNodes.size() == 1);
    assert(substitutionLastOpNodes.size() == 1);
    assert(*substitutionFirstOpNodes.begin() == *substitutionLastOpNodes.begin());
    auto resize3c3pOpNode = *substitutionFirstOpNodes.begin();
    assert(substituteGraph.metadata(resize3c3pOpNode).get<cv::gimpl::Op>().k.name == "test.resize3c3p");

    assert(patternFirstOpNodes.size() == 1);
    assert(patternLastOpNodes.size() == 1);
    assert(*patternFirstOpNodes.begin() != *patternLastOpNodes.begin());
    auto resizeOpNode = *patternFirstOpNodes.begin();
    assert(patternGraph.metadata(resizeOpNode).get<cv::gimpl::Op>().k.name == "org.opencv.core.transform.resize");
    auto toNCHWOpNode = *patternLastOpNodes.begin();
    assert(patternGraph.metadata(toNCHWOpNode).get<cv::gimpl::Op>().k.name == "test.toNCHW");

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> firstSubstituteOpNodesMatches;
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> lastSubstituteOpNodesMatches;

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> firstSubstituteDataNodesMatches;
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> lastSubstituteDataNodesMatches;

    firstSubstituteOpNodesMatches[resizeOpNode] = resize3c3pOpNode;
    lastSubstituteOpNodesMatches[toNCHWOpNode] = resize3c3pOpNode;


    assert(resizeOpNode->inNodes().size() == 1);
    assert(toNCHWOpNode->outNodes().size() == 1);

    assert(resize3c3pOpNode->inNodes().size() == 1);
    assert(resize3c3pOpNode->outNodes().size() == 1);

    firstSubstituteDataNodesMatches[*resizeOpNode->inNodes().begin()] = *resize3c3pOpNode->inNodes().begin();
    lastSubstituteDataNodesMatches[*toNCHWOpNode->outNodes().begin()] = *resize3c3pOpNode->outNodes().begin();

    cv::gimpl::SubgraphMatch substituteMatch{ };
    substituteMatch.inputDataNodesMatches = firstSubstituteDataNodesMatches;
    substituteMatch.firstOpNodesMatches = firstSubstituteOpNodesMatches;
    substituteMatch.lastOpNodesMatches = lastSubstituteOpNodesMatches;
    substituteMatch.outputDataNodesMatches = lastSubstituteDataNodesMatches;

    substituteMatches(patternGraph, compGraph, match, substituteGraph, substituteMatch);
    //---------------------------------------------------------------


    //----------------Substituted graph recompilation----------------

    //-------------------Only for review purpose---------------------
    //---------------------Not production code-----------------------
    //Temporary tricks before the code relocation to the required place.

    //--------Bad and UB trick to call protected method:-------------
    class AdeGraphTrick : public cv::gimpl::GModel::ConstGraph {
    public:
        using cv::gimpl::GModel::ConstGraph::getCGraph;
    };

    auto constCompGraph = static_cast<cv::gimpl::GModel::ConstGraph>(compGraph);
    auto& constCompGraphRef = constCompGraph;

    auto& constCompAdeGraph = static_cast<AdeGraphTrick &>(constCompGraphRef).getCGraph();
    auto& compAdeGraph = const_cast<ade::Graph&>(constCompAdeGraph);
    //------------------End of the Bad and UB trick.-----------------

    std::unique_ptr<ade::Graph> adeGraphPtr(&compAdeGraph);
    // TODO FIX: also for resource sharing
    std::unique_ptr<cv::gimpl::GCompiler> compiler(new cv::gimpl::GCompiler(*computation, cv::descr_of(cv::gin(testY, testUV)), cv::compile_args(substitutePkg)));
    compiler->runPasses(*adeGraphPtr);
    compiler->compileIslands(*adeGraphPtr);
    auto compiled = compiler->produceCompiled(std::move(adeGraphPtr));
    //---------------------------------------------------------------

    //------------------Substituted graph testing--------------------
    cv::Mat testPlanarImForFusedGraph;
    compiled(gin(testY, testUV), gout(testPlanarImForFusedGraph));

    EXPECT_TRUE(AbsExact()(testPlanarIm, testPlanarImForFusedGraph));
    //--------------------------------------------------------------

    // Leave memory leaks here to avoid crashes:)
    compiler.release();
    adeGraphPtr.release();
    computation.release();
}
#endif
} // namespace opencv_test
