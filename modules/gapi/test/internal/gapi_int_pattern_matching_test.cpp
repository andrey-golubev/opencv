// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <stdexcept>

#include <opencv2/gapi/gtransform.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include "compiler/gmodel.hpp"
#include "compiler/gmodel_priv.hpp"

#include "api/gcomputation_priv.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/passes/passes.hpp"

#include "compiler/passes/pattern_matching.hpp"

#include "../common/gapi_tests_common.hpp"

#include "logger.hpp"

namespace opencv_test
{

namespace matching_test {
namespace  {
using V = std::vector<ade::NodeHandle>;
using S =  std::unordered_set< ade::NodeHandle
                             , ade::HandleHasher<ade::Node>
                             >;

void initGModel(ade::Graph& gr,
                cv::GProtoInputArgs&& in,
                cv::GProtoOutputArgs&& out) {

    cv::gimpl::GModel::Graph gm(gr);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(gr)
            .put(in.m_args, out.m_args);

    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    gm.metadata().set(p);
}

bool isConsumedBy(cv::gimpl::GModel::Graph gm, ade::NodeHandle data_nh, ade::NodeHandle op_nh) {
    auto oi = cv::gimpl::GModel::orderedInputs(gm, op_nh);
    return std::find(oi.begin(), oi.end(), data_nh) != oi.end();
}

std::string opName(cv::gimpl::GModel::Graph gm, ade::NodeHandle op_nh) {
    return gm.metadata(op_nh).get<cv::gimpl::Op>().k.name;
}

}
} // matching_test

TEST(PatternMatching, TestFuncDoesNotChangeTestGraph)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
    }

    // Test
    ade::Graph tg;
    GMat in;
    GMat out = cv::gapi::bitwise_not(in);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    matching_test::S nodes{ tgm.nodes().begin(), tgm.nodes().end() };

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);

    auto input_data_nhs = tgm.metadata().get<cv::gimpl::Protocol>().in_nhs;
    auto output_data_nhs = tgm.metadata().get<cv::gimpl::Protocol>().out_nhs;

    EXPECT_EQ(1u, input_data_nhs.size());
    EXPECT_EQ(1u, output_data_nhs.size());
    EXPECT_EQ(in_nh, *input_data_nhs.begin());
    EXPECT_EQ(out_nh, *output_data_nhs.begin());
    EXPECT_EQ(0u, in_nh->inEdges().size());
    EXPECT_EQ(0u, out_nh->outEdges().size());
    EXPECT_EQ(1u, in_nh->outEdges().size());
    EXPECT_EQ(1u, out_nh->inEdges().size());

    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh); //bitwise_not
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_EQ(1u, op_nh->inEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(1u, op_nh->outEdges().size());
}

TEST(PatternMatching, TestSimple1)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat out = cv::gapi::bitwise_not(in);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({in_nh, out_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{out_nh}, match.protoOuts());
}

TEST(PatternMatching, TestSimple2)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp = cv::gapi::bitwise_not(in);
    GMat out = cv::gapi::blur(tmp, cv::Size(3, 3));
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, tmp_nh);

    EXPECT_EQ(matching_test::S({in_nh, tmp_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{tmp_nh}, match.protoOuts());
}

TEST(PatternMatching, TestSimple3)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp = cv::gapi::blur(in, cv::Size(3, 3));
    GMat out = cv::gapi::bitwise_not(tmp);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({tmp_nh, out_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{tmp_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{out_nh}, match.protoOuts());
}

TEST(PatternMatching, TestMultiplePatternOuts)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat dx, dy;
        std::tie(dx, dy) = cv::gapi::SobelXY(in, -1, 1);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(dx, dy));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat dx, dy;
    std::tie(dx, dy) = cv::gapi::SobelXY(in, -1, 1);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(dx, dy));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(4u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto dx_nh = cv::gimpl::GModel::dataNodeOf(tgm, dx);
    const auto dy_nh = cv::gimpl::GModel::dataNodeOf(tgm, dy);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, dx_nh);
    EXPECT_EQ(op_nh,  cv::gimpl::GModel::producerOf(tgm, dy_nh));

    EXPECT_EQ(matching_test::S({in_nh, dx_nh, dy_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::imgproc::GSobelXY::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V({dx_nh, dy_nh}), match.protoOuts());
}

TEST(PatternMatching, TestPrepResizeSplit3)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::resize(in, cv::Size{224, 224});
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(tmp);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(b, g, r));
    }

    // Test
    ade::Graph tg;
    GMat y, uv;
    GMat bgr = cv::gapi::NV12toBGR(y, uv);
    GMat tmp = cv::gapi::resize(bgr, cv::Size{224, 224});
    GMat b, g, r;
    std::tie(b, g, r) = cv::gapi::split3(tmp);
    matching_test::initGModel(tg, cv::GIn(y, uv), cv::GOut(b, g, r));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(7u, nodes.size());

    const auto bgr_nh = cv::gimpl::GModel::dataNodeOf(tgm, bgr);
    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto b_nh = cv::gimpl::GModel::dataNodeOf(tgm, b);
    const auto g_nh = cv::gimpl::GModel::dataNodeOf(tgm, g);
    const auto r_nh = cv::gimpl::GModel::dataNodeOf(tgm, r);

    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, tmp_nh); // 1st resize
    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, b_nh); // 2nd split3
    EXPECT_EQ(op2_nh, cv::gimpl::GModel::producerOf(tgm, g_nh));
    EXPECT_EQ(op2_nh, cv::gimpl::GModel::producerOf(tgm, r_nh));

    EXPECT_EQ(matching_test::S({bgr_nh, tmp_nh,   b_nh,   g_nh,
                                          r_nh, op1_nh, op2_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::core::GResize::id(), matching_test::opName(tgm, op1_nh));
    EXPECT_EQ(cv::gapi::core::GSplit3::id(), matching_test::opName(tgm, op2_nh));

    EXPECT_EQ(1u, tmp_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, bgr_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp_nh, op2_nh));

    EXPECT_EQ(matching_test::S{ op1_nh }, match.startOps());
    EXPECT_EQ(matching_test::S{ op2_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V{ bgr_nh }, match.protoIns());
    EXPECT_EQ(matching_test::V({ b_nh, g_nh, r_nh }), match.protoOuts());
}

G_TYPED_KERNEL(GToNCHW, <GMatP(GMat)>, "test.toNCHW") {
    static GMatDesc outMeta(GMatDesc in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar == false);
        return in.asPlanar();
    }
};

static GMatP toNCHW(const GMat& src)
{
    return GToNCHW::on(src);
}

TEST(PatternMatching, TestPrepResizeToNCHW)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::resize(in, cv::Size{224, 224});
        GMatP plr = toNCHW(tmp);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(plr));
    }

    // Test
    ade::Graph tg;
    GMat y, uv;
    GMat bgr = cv::gapi::NV12toBGR(y, uv);
    GMat tmp = cv::gapi::resize(bgr, cv::Size{224, 224});
    GMatP plr = toNCHW(tmp);
    matching_test::initGModel(tg, cv::GIn(y, uv), cv::GOut(plr));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(5u, nodes.size());

    const auto bgr_nh = cv::gimpl::GModel::dataNodeOf(tgm, bgr);
    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto plr_nh = cv::gimpl::GModel::dataNodeOf(tgm, plr);

    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, tmp_nh); // 1st resize
    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, plr_nh); // 2nd toNCHW

    EXPECT_EQ(matching_test::S({bgr_nh, tmp_nh, plr_nh, op1_nh, op2_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::core::GResize::id(), matching_test::opName(tgm, op1_nh));
    EXPECT_EQ(GToNCHW::id(), matching_test::opName(tgm, op2_nh));

    EXPECT_EQ(1u, tmp_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, bgr_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp_nh, op2_nh));

    EXPECT_EQ(matching_test::S{ op1_nh }, match.startOps());
    EXPECT_EQ(matching_test::S{ op2_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V{ bgr_nh }, match.protoIns());
    EXPECT_EQ(matching_test::V{ plr_nh }, match.protoOuts());
}

TEST(PatternMatching, TestPrepNV12toBGRToNCHW)
{
    // Pattern
    ade::Graph pg;
    {
        GMat y, uv;
        GMat bgr = cv::gapi::NV12toBGR(y, uv);
        GMatP plr = toNCHW(bgr);
        matching_test::initGModel(pg, cv::GIn(y, uv), cv::GOut(plr));
    }

    // Test
    ade::Graph tg;
    GMat y, uv;
    GMat bgr = cv::gapi::NV12toBGR(y, uv);
    GMatP plr = toNCHW(bgr);
    GMat rsz = cv::gapi::resizeP(plr, cv::Size{224, 224});
    matching_test::initGModel(tg, cv::GIn(y, uv), cv::GOut(rsz));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(6u, nodes.size());

    const auto y_nh = cv::gimpl::GModel::dataNodeOf(tgm, y);
    const auto uv_nh = cv::gimpl::GModel::dataNodeOf(tgm, uv);
    const auto bgr_nh = cv::gimpl::GModel::dataNodeOf(tgm, bgr);
    const auto plr_nh = cv::gimpl::GModel::dataNodeOf(tgm, plr);

    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, bgr_nh); // 1st NV12toBGR
    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, plr_nh); // 2nd toNCHW

    EXPECT_EQ(matching_test::S({y_nh, uv_nh, bgr_nh, plr_nh, op1_nh, op2_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::imgproc::GNV12toBGR::id(), matching_test::opName(tgm, op1_nh));
    EXPECT_EQ(GToNCHW::id(), matching_test::opName(tgm, op2_nh));

    EXPECT_EQ(1u, bgr_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, y_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, uv_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, bgr_nh, op2_nh));

    EXPECT_EQ(matching_test::S{ op1_nh }, match.startOps());
    EXPECT_EQ(matching_test::S{ op2_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V({ y_nh, uv_nh }), match.protoIns());
    EXPECT_EQ(matching_test::V{ plr_nh }, match.protoOuts());
}

//FIXME: To switch from filter2d kernel (which shall be matched by params too) to another one
TEST(PatternMatching, MatchChainInTheMiddle)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::filter2D(in, -1, {});
        GMat out = cv::gapi::filter2D(tmp, -1, {});
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp1 = cv::gapi::erode3x3(in);
    GMat tmp2 = cv::gapi::filter2D(tmp1, -1, {});
    GMat tmp3 = cv::gapi::filter2D(tmp2, -1, {});
    GMat out = cv::gapi::dilate3x3(tmp3);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(5u, nodes.size());

    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp1);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp2);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp3);
    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, tmp2_nh); // 1st filter2D
    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, tmp3_nh); // 2nd filter2D

    EXPECT_EQ(matching_test::S({tmp1_nh, tmp2_nh, tmp3_nh, op1_nh, op2_nh}), nodes);

    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op1_nh));
    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op2_nh));

    EXPECT_EQ(1u, tmp2_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp1_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp2_nh, op2_nh));

    EXPECT_EQ(matching_test::S({op1_nh}), match.startOps());
    EXPECT_EQ(matching_test::S({op2_nh}), match.finishOps());
    EXPECT_EQ(matching_test::V{ tmp1_nh }, match.protoIns());
    EXPECT_EQ(matching_test::V{ tmp3_nh }, match.protoOuts());
}

TEST(PatternMatching, TestMultipleStartOps1)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2;
        GMat er = cv::gapi::erode3x3(in1);
        GMat dil = cv::gapi::dilate3x3(in2);
        GMat out = cv::gapi::add(er, dil);
        matching_test::initGModel(pg, cv::GIn(in1, in2), cv::GOut(out));
    }

    // Test
    ade::Graph tg;

    GMat in1, in2, in3, in4, in5, in6;
    GMat er1 = cv::gapi::erode3x3(in1);
    GMat er2 = cv::gapi::erode3x3(in2);
    GMat er3 = cv::gapi::erode3x3(in3);
    GMat er4 = cv::gapi::erode3x3(in4);
    GMat dil1 = cv::gapi::dilate3x3(in5);
    GMat dil2 = cv::gapi::dilate3x3(in6);
    GMat out1 = cv::gapi::add(er1, er2);
    GMat out2 = cv::gapi::add(er3, dil2);
    matching_test::initGModel(tg, cv::GIn(in1, in2, in3, in4, in5, in6), cv::GOut(out1, out2, er4, dil1));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(8u, nodes.size());

    const auto in3_nh = cv::gimpl::GModel::dataNodeOf(tgm, in3);
    const auto in6_nh = cv::gimpl::GModel::dataNodeOf(tgm, in6);
    const auto er3_nh = cv::gimpl::GModel::dataNodeOf(tgm, er3);
    const auto dil2_nh = cv::gimpl::GModel::dataNodeOf(tgm, dil2);
    const auto out2_nh = cv::gimpl::GModel::dataNodeOf(tgm, out2);

    const auto er_op_nh = cv::gimpl::GModel::producerOf(tgm, er3_nh);
    const auto dil_op_nh = cv::gimpl::GModel::producerOf(tgm, dil2_nh);
    const auto add_op_nh = cv::gimpl::GModel::producerOf(tgm, out2_nh);

    EXPECT_EQ(matching_test::S({in3_nh, in6_nh, er3_nh, dil2_nh, out2_nh,
                                er_op_nh, dil_op_nh, add_op_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::imgproc::GErode::id(), matching_test::opName(tgm, er_op_nh));
    EXPECT_EQ(cv::gapi::imgproc::GDilate::id(), matching_test::opName(tgm, dil_op_nh));
    EXPECT_EQ(cv::gapi::core::GAdd::id(), matching_test::opName(tgm, add_op_nh));

    EXPECT_EQ(1u, er3_nh->outEdges().size());
    EXPECT_EQ(1u, dil2_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in3_nh, er_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in6_nh, dil_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, er3_nh, add_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, dil2_nh, add_op_nh));

    EXPECT_EQ(matching_test::S({ er_op_nh, dil_op_nh }), match.startOps());
    EXPECT_EQ(matching_test::S{ add_op_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V({ in3_nh, in6_nh }), match.protoIns());
    EXPECT_EQ(matching_test::V{ out2_nh }, match.protoOuts());
}

TEST(PatternMatching, TestMultipleStartOps2)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2;
        GMat er = cv::gapi::erode3x3(in1);
        GMat dil = cv::gapi::dilate3x3(in2);
        GMat out = cv::gapi::add(er, dil);
        matching_test::initGModel(pg, cv::GIn(in1, in2), cv::GOut(out));
    }

    // Test
    ade::Graph tg;

    GMat in1, in2;
    GMat er = cv::gapi::erode3x3(in1);
    GMat dil1 = cv::gapi::dilate3x3(in2);
    GMat dil2 = cv::gapi::dilate3x3(dil1);
    GMat out = cv::gapi::add(er, dil2);
    matching_test::initGModel(tg, cv::GIn(in1, in2), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(8u, nodes.size());

    const auto in1_nh = cv::gimpl::GModel::dataNodeOf(tgm, in1);
    const auto dil1_nh = cv::gimpl::GModel::dataNodeOf(tgm, dil1);
    const auto er_nh = cv::gimpl::GModel::dataNodeOf(tgm, er);
    const auto dil2_nh = cv::gimpl::GModel::dataNodeOf(tgm, dil2);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);

    const auto er_op_nh = cv::gimpl::GModel::producerOf(tgm, er_nh);
    const auto dil_op_nh = cv::gimpl::GModel::producerOf(tgm, dil2_nh);
    const auto add_op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({in1_nh, dil1_nh, er_nh, dil2_nh, out_nh,
                                er_op_nh, dil_op_nh, add_op_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::imgproc::GErode::id(), matching_test::opName(tgm, er_op_nh));
    EXPECT_EQ(cv::gapi::imgproc::GDilate::id(), matching_test::opName(tgm, dil_op_nh));
    EXPECT_EQ(cv::gapi::core::GAdd::id(), matching_test::opName(tgm, add_op_nh));

    EXPECT_EQ(1u, er_nh->outEdges().size());
    EXPECT_EQ(1u, dil2_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in1_nh, er_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, dil1_nh, dil_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, er_nh, add_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, dil2_nh, add_op_nh));

    EXPECT_EQ(matching_test::S({ er_op_nh, dil_op_nh }), match.startOps());
    EXPECT_EQ(matching_test::S{ add_op_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V({ in1_nh, dil1_nh }), match.protoIns());
    EXPECT_EQ(matching_test::V{ out_nh }, match.protoOuts());
}

TEST(PatternMatching, TestInexactMatchOfInOutData)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::dilate3x3(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
    }

    // Test
    ade::Graph tg;
    GMat in;
    GMat out1 = cv::gapi::erode3x3(in);
    GMat out2 = cv::gapi::boxFilter(in, -1, cv::Size(3, 3));
    GMat tmp = cv::gapi::dilate3x3(in);
    GScalar out3 = cv::gapi::sum(tmp);
    GScalar out4 = cv::gapi::mean(tmp);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out1, out2, out3, out4));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);

    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, tmp_nh); // dilate3x3

    EXPECT_EQ(matching_test::S({in_nh, tmp_nh, op_nh}),
              nodes);

    EXPECT_EQ(cv::gapi::imgproc::GDilate::id(), matching_test::opName(tgm, op_nh));

    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));


    EXPECT_EQ(matching_test::S{ op_nh }, match.startOps());
    EXPECT_EQ(matching_test::S{ op_nh }, match.finishOps());
    EXPECT_EQ(matching_test::V{ in_nh }, match.protoIns());
    EXPECT_EQ(matching_test::V{ tmp_nh }, match.protoOuts());

    EXPECT_GT(in_nh->outEdges().size(), 1u);
    EXPECT_GT(tmp_nh->outEdges().size(), 1u);
}

//FIXME: The start ops matching shall be reworked to more smarter way.
// Start ops matching shall get rid of non valid matchings sample,
// where two identical start ops in the pattern refer to the only one in the test.
TEST(PatternMatching, TestManySameStartOpsAndHinge)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2, in3;
        GMat er1 = cv::gapi::erode3x3(in1);
        GMat er2 = cv::gapi::erode3x3(in2);
        GMat er3 = cv::gapi::erode3x3(in3);
        GMat mrg = cv::gapi::merge3(er1, er2, er3);
        matching_test::initGModel(pg, cv::GIn(in1, in2, in3), cv::GOut(mrg));
    }

    // Test
    ade::Graph tg;
    GMat in1, in2, in3;
    GMat er1 = cv::gapi::erode3x3(in1);
    GMat er2 = cv::gapi::erode3x3(in2);
    GMat er3 = cv::gapi::erode3x3(in3);
    GMat mrg = cv::gapi::merge3(er1, er2, er3);
    matching_test::initGModel(tg, cv::GIn(in1, in2, in3), cv::GOut(mrg));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(11u, nodes.size());
    EXPECT_EQ(matching_test::S(tgm.nodes().begin(), tgm.nodes().end()),
              nodes);
}

//FIXME: The start ops matching shall be reworked to more smarter way.
// Start ops matching shall get rid of non valid matchings sample,
// where two identical start ops in the pattern refer to the only one in the test.
TEST(PatternMatching, TestManySameStartOpsAndHinge2)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2, in3;
        GMat er1 = cv::gapi::erode3x3(in1);
        GMat er2 = cv::gapi::erode3x3(in2);
        GMat er3 = cv::gapi::erode3x3(in3);
        GMat dil1 = cv::gapi::dilate3x3(er1);
        GMat dil2 = cv::gapi::dilate3x3(er2);
        GMat dil3 = cv::gapi::dilate3x3(er3);
        GMat mrg = cv::gapi::merge3(dil1, dil2, dil3);
        matching_test::initGModel(pg, cv::GIn(in1, in2, in3), cv::GOut(mrg));
    }

    // Test
    ade::Graph tg;
    GMat in1, in2, in3;
    GMat er1 = cv::gapi::erode3x3(in1);
    GMat er2 = cv::gapi::erode3x3(in2);
    GMat er3 = cv::gapi::erode3x3(in3);
    GMat dil1 = cv::gapi::dilate3x3(er1);
    GMat dil2 = cv::gapi::dilate3x3(er2);
    GMat dil3 = cv::gapi::dilate3x3(er3);
    GMat mrg = cv::gapi::merge3(dil1, dil2, dil3);
    matching_test::initGModel(tg, cv::GIn(in1, in2, in3), cv::GOut(mrg));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(17u, nodes.size());
    EXPECT_EQ(matching_test::S(tgm.nodes().begin(), tgm.nodes().end()),
              nodes);
}

//FIXME: The start ops matching shall be reworked to more smarter way.
// Start ops matching shall get rid of non valid matchings sample,
// where two identical start ops in the pattern refer to the only one in the test.
TEST(PatternMatching, TestTwoChainsOnTheHingeIsomorphism)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2;
        GMat er1 = cv::gapi::erode3x3(in1);
        GMat er2 = cv::gapi::erode3x3(in2);
        GMat mdb = cv::gapi::medianBlur(er1, 3);
        GMat gb = cv::gapi::gaussianBlur(er2, cv::Size(5, 5), 0.12);
        GMat conc = cv::gapi::concatVert(mdb, gb);
        matching_test::initGModel(pg, cv::GIn(in1, in2), cv::GOut(conc));
    }

    // Test
    ade::Graph tg;
    GMat in1, in2;
    GMat er1 = cv::gapi::erode3x3(in1);
    GMat er2 = cv::gapi::erode3x3(in2);
    GMat gb = cv::gapi::gaussianBlur(er1, cv::Size(5, 5), 0.12);
    GMat mdb = cv::gapi::medianBlur(er2, 3);
    GMat conc = cv::gapi::concatVert(mdb, gb);
    matching_test::initGModel(tg, cv::GIn(in1, in2), cv::GOut(conc));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(12u, nodes.size());
    EXPECT_EQ(matching_test::S(tgm.nodes().begin(), tgm.nodes().end()),
              nodes);

    const auto in1_nh = cv::gimpl::GModel::dataNodeOf(tgm, in1);
    const auto in2_nh = cv::gimpl::GModel::dataNodeOf(tgm, in2);

    EXPECT_EQ(matching_test::V({ in2_nh, in1_nh }), match.protoIns());
}

TEST(PatternMatching, TestPatternHasMoreInDataNodes)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2, in3;
        GMat out = cv::gapi::merge3(in1, in2, in3);
        matching_test::initGModel(pg, cv::GIn(in1, in2, in3), cv::GOut(out));
    }

    // Test
    ade::Graph tg;
    GMat in;
    GMat out = cv::gapi::merge3(in, in, in);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());
    EXPECT_EQ(matching_test::S(tgm.nodes().begin(), tgm.nodes().end()),
              nodes);

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);

    EXPECT_EQ(matching_test::V({ in_nh, in_nh, in_nh }), match.protoIns());
}

TEST(PatternMatching, TestPatternHasFewerInDataNodes)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::merge3(in, in, in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
    }

    // Test
    ade::Graph tg;
    GMat in1, in2, in3;
    GMat out = cv::gapi::merge3(in1, in2, in3);
    matching_test::initGModel(tg, cv::GIn(in1, in2, in3), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_FALSE(match.ok());
}

TEST(PatternMatching, TestTwoMatchingsOneCorrect)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in1, in2;
        GMat n = cv::gapi::bitwise_not(in1);
        GMat e = cv::gapi::erode3x3(in1);
        GMat d = cv::gapi::dilate3x3(in2);
        GMat out = cv::gapi::merge3(n, e, d);
        matching_test::initGModel(pg, cv::GIn(in1, in2), cv::GOut(out));
    }

    // Test
    ade::Graph tg;
    GMat in1, in2;
    GMat n = cv::gapi::bitwise_not(in1);
    GMat e = cv::gapi::erode3x3(in2);
    GMat d = cv::gapi::dilate3x3(in2);
    GMat mrg = cv::gapi::merge3(n, e, d);
    GMat i, sqi;
    std::tie(i, sqi) = cv::gapi::integral(mrg);
    GMat n1 = cv::gapi::bitwise_not(i);
    GMat e1 = cv::gapi::erode3x3(i);
    GMat d1 = cv::gapi::dilate3x3(sqi);
    GMat out = cv::gapi::merge3(n1, e1, d1);
    matching_test::initGModel(tg, cv::GIn(in1, in2), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(10u, nodes.size());

    const auto i_nh = cv::gimpl::GModel::dataNodeOf(tgm, i);
    const auto sqi_nh = cv::gimpl::GModel::dataNodeOf(tgm, sqi);
    const auto n1_nh = cv::gimpl::GModel::dataNodeOf(tgm, n1);
    const auto e1_nh = cv::gimpl::GModel::dataNodeOf(tgm, e1);
    const auto d1_nh = cv::gimpl::GModel::dataNodeOf(tgm, d1);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);

    const auto n_op_nh = cv::gimpl::GModel::producerOf(tgm, n1_nh);
    const auto e_op_nh = cv::gimpl::GModel::producerOf(tgm, e1_nh);
    const auto d_op_nh = cv::gimpl::GModel::producerOf(tgm, d1_nh);
    const auto m_op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({i_nh, sqi_nh, n1_nh, e1_nh, d1_nh, out_nh,
                                n_op_nh, e_op_nh, d_op_nh, m_op_nh}), nodes);

    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, n_op_nh));
    EXPECT_EQ(cv::gapi::imgproc::GErode::id(), matching_test::opName(tgm, e_op_nh));
    EXPECT_EQ(cv::gapi::imgproc::GDilate::id(), matching_test::opName(tgm, d_op_nh));
    EXPECT_EQ(cv::gapi::core::GMerge3::id(), matching_test::opName(tgm, m_op_nh));

    EXPECT_TRUE(matching_test::isConsumedBy(tgm, i_nh, n_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, i_nh, e_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, sqi_nh, d_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, n1_nh, m_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, e1_nh, m_op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, d1_nh, m_op_nh));
    EXPECT_EQ(1u, n1_nh->outEdges().size());
    EXPECT_EQ(1u, e1_nh->outEdges().size());
    EXPECT_EQ(1u, d1_nh->outEdges().size());

    EXPECT_EQ(matching_test::S({n_op_nh, e_op_nh, d_op_nh}), match.startOps());
    EXPECT_EQ(matching_test::S{m_op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V({i_nh, sqi_nh}), match.protoIns());
    EXPECT_EQ(matching_test::V{out_nh}, match.protoOuts());}

TEST(PatternMatching, CheckNoMatch)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::filter2D(in, -1, {});
        GMat out = cv::gapi::filter2D(tmp, -1, {});
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    {
        GMat in;
        GMat tmp1 = cv::gapi::erode3x3(in);
        GMat out = cv::gapi::dilate3x3(tmp1);
        matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));
    }

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_FALSE(match.ok());
}

TEST(PatternMatching, adeSmokeTest)
{
    ade::Graph g;
    ade::NodeHandle src = g.createNode();
    ade::NodeHandle dst = g.createNode();
    g.link(src, dst);
    g.link(src, dst);

    EXPECT_EQ(2u, dst->inNodes().size());
}

// --------------------------------------------------------------------------------------
// Unit tests for matching pattern to substitute
using GraphCreator = std::function<std::tuple<
    std::unique_ptr<ade::Graph>,
    std::vector<ade::NodeHandle>,
    std::vector<ade::NodeHandle>>()>;
struct PatternMatchingMatchPatternToSubsitute :
    TestWithParam<std::tuple<GraphCreator, GraphCreator>> {};
TEST_P(PatternMatchingMatchPatternToSubsitute, TestMatchFound)
{
    std::unique_ptr<ade::Graph> pg;  // pattern
    std::vector<ade::NodeHandle> p_ins, p_outs;  // input/output nodes of pattern

    std::unique_ptr<ade::Graph> sg;  // substitute
    std::vector<ade::NodeHandle> s_ins, s_outs;  // input/output nodes of substitute

    // initialize
    std::forward_as_tuple(std::tie(pg, p_ins, p_outs), std::tie(sg, s_ins, s_outs)) =
        std::make_tuple(std::get<0>(GetParam())(), std::get<1>(GetParam())());

    // match
    cv::gimpl::GModel::Graph pgm(*pg);
    cv::gimpl::GModel::Graph sgm(*sg);
    auto match = cv::gimpl::matchPatternToSubstitute(pgm, sgm,
        pgm.metadata().get<cv::gimpl::Protocol>(), sgm.metadata().get<cv::gimpl::Protocol>());

    // check the result
    EXPECT_TRUE(match.partialOk());
    EXPECT_FALSE(match.ok());

    const auto verify_nodes = [] (const std::vector<ade::NodeHandle>& p_nodes,
        const std::vector<ade::NodeHandle>& s_nodes,
        const cv::gimpl::SubgraphMatch::M& matched_nodes) {
        // check that sizes are equal
        EXPECT_EQ(p_nodes.size(), matched_nodes.size());
        EXPECT_EQ(s_nodes.size(), matched_nodes.size());
        for (size_t i = 0; i < p_nodes.size(); ++i) {
            const auto& p_node = p_nodes[i];
            const auto& s_node = s_nodes[i];

            auto found = matched_nodes.find(p_node);
            ASSERT_NE(matched_nodes.cend(), found);  // assert - cannot de-reference end()
            EXPECT_EQ(s_node, found->second);
        }
    };

    verify_nodes(p_ins, s_ins, match.inputDataNodes);
    verify_nodes(p_outs, s_outs, match.outputDataNodes);
}

struct PatternMatchingMatchPatternToSubsituteBadArg :
    TestWithParam<std::tuple<GraphCreator, GraphCreator>> {};
TEST_P(PatternMatchingMatchPatternToSubsituteBadArg, TestMatchNotFound)
{
    std::unique_ptr<ade::Graph> pg;  // pattern
    std::vector<ade::NodeHandle> p_ins, p_outs;  // input/output nodes of pattern

    std::unique_ptr<ade::Graph> sg;  // substitute
    std::vector<ade::NodeHandle> s_ins, s_outs;  // input/output nodes of substitute

    // initialize
    std::forward_as_tuple(std::tie(pg, p_ins, p_outs), std::tie(sg, s_ins, s_outs)) =
        std::make_tuple(std::get<0>(GetParam())(), std::get<1>(GetParam())());

    // match
    cv::gimpl::GModel::Graph pgm(*pg);
    cv::gimpl::GModel::Graph sgm(*sg);
    auto match = cv::gimpl::matchPatternToSubstitute(pgm, sgm,
        pgm.metadata().get<cv::gimpl::Protocol>(), sgm.metadata().get<cv::gimpl::Protocol>());

    // FIXME: check anything else?
    EXPECT_FALSE(match.partialOk());
    EXPECT_FALSE(match.ok());
    EXPECT_TRUE(match.empty());
}

INSTANTIATE_TEST_CASE_P(OneInputOneOutput, PatternMatchingMatchPatternToSubsitute,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in;
                    GMat out = cv::gapi::bitwise_not(in);
                    matching_test::initGModel(*pg, cv::GIn(in), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in;
                    GMat tmp = cv::gapi::bitwise_not(in);
                    GMat out = cv::gapi::add(in, tmp);
                    matching_test::initGModel(*sg, cv::GIn(in), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

INSTANTIATE_TEST_CASE_P(ManyInputsOneOutput, PatternMatchingMatchPatternToSubsitute,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1, in2;
                    GMat out = cv::gapi::add(in1, in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in1, in2;
                    GMat tmp = cv::gapi::bitwise_or(in1, in2);
                    GMat out = cv::gapi::add(in1, tmp);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

INSTANTIATE_TEST_CASE_P(OneInputManyOutputs, PatternMatchingMatchPatternToSubsitute,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in;
                    GMat out1 = cv::gapi::add(in, in);
                    GMat out2 = cv::gapi::mul(in, in);
                    matching_test::initGModel(*pg, cv::GIn(in), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in;
                    GMat tmp = cv::gapi::bitwise_not(in);
                    GMat out1 = cv::gapi::add(in, tmp);
                    GMat out2 = cv::gapi::bitwise_xor(in, tmp);
                    matching_test::initGModel(*sg, cv::GIn(in), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

G_TYPED_KERNEL(GDummyArrayKernel, <GArray<int>(GArray<int>)>, "test.dummy_array_kernel") {
    static GArrayDesc outMeta(GArrayDesc in) { return in; }
};
GAPI_OCV_KERNEL(GOCVDummyArrayKernel, GDummyArrayKernel) {
    static void run(const std::vector<int>& in, std::vector<int>& out) { out = in; }
};

INSTANTIATE_TEST_CASE_P(ManyInputsManyOutputs, PatternMatchingMatchPatternToSubsitute,
    Values(
        std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1, in2;
                    GScalar in3;
                    GMat out1 = cv::gapi::addC(in1, in3);
                    GMat unused = cv::gapi::cmpEQ(in2, in3); (void)unused;
                    GMat out2 = cv::gapi::add(in1, in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in1, in2;
                    GScalar in3;
                    GMat tmp = cv::gapi::bitwise_not(in1);
                    GMat out1 = cv::gapi::mulC(cv::gapi::add(in1, tmp), in3);
                    GMat out2 = cv::gapi::divC(in2, in3, 0.0);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            }),
        std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1, in2;
                    GScalar in3;
                    GScalar out1 = cv::gapi::mean(in2);
                    GMat out2 = cv::gapi::addC(in1, in3);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in1, in2;
                    GScalar in3;
                    GMat tmp = cv::gapi::bitwise_not(in1);
                    GScalar out1 = cv::gapi::normL1(cv::gapi::divC(in2, in3, 0.0));
                    GMat unused = cv::gapi::concatHor(tmp, in1); (void)unused;
                    GMat out2 = cv::gapi::mulC(cv::gapi::add(in1, tmp), in3);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            }),
        std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1;
                    GArray<int> in2;
                    GScalar in3;
                    GScalar out1 = cv::gapi::mean(in1);
                    GMat out2 = cv::gapi::addC(in1, in3);
                    GArray<int> out3 = GDummyArrayKernel::on(in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2, out3));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2.strip()));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out3.strip()));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in1;
                    GArray<int> in2;
                    GScalar in3;
                    GMat tmp = cv::gapi::bitwise_not(in1);
                    GScalar out1 = cv::gapi::normL1(cv::gapi::divC(in1, in3, 0.0));
                    GMat out2 = cv::gapi::mulC(cv::gapi::add(in1, tmp), in3);
                    GArray<int> out3 = GDummyArrayKernel::on(in2);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2, out3));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2.strip()));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out3.strip()));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

// BadArg instantiations
INSTANTIATE_TEST_CASE_P(OneInputOneOutput, PatternMatchingMatchPatternToSubsituteBadArg,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in;
                    GMat out = cv::gapi::bitwise_not(in);
                    matching_test::initGModel(*pg, cv::GIn(in), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GArray<int> in;
                    GArray<int> out = GDummyArrayKernel::on(in);
                    matching_test::initGModel(*sg, cv::GIn(in), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in.strip()));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out.strip()));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

INSTANTIATE_TEST_CASE_P(ManyInputsOneOutput, PatternMatchingMatchPatternToSubsituteBadArg,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1;
                    GScalar in2;
                    GMat out = cv::gapi::addC(in1, in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GScalar in1;
                    GMat in2;
                    GMat out = cv::gapi::addC(in2, in1);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

INSTANTIATE_TEST_CASE_P(OneInputManyOutputs, PatternMatchingMatchPatternToSubsituteBadArg,
    Values(std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in;
                    GMat out1 = cv::gapi::add(in, in);
                    GScalar out2 = cv::gapi::mean(in);
                    matching_test::initGModel(*pg, cv::GIn(in), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in;
                    GMat tmp = cv::gapi::bitwise_not(in);
                    GScalar out1 = cv::gapi::sum(tmp);
                    GMat out2 = cv::gapi::bitwise_xor(in, tmp);
                    matching_test::initGModel(*sg, cv::GIn(in), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

INSTANTIATE_TEST_CASE_P(ManyInputsManyOutputs, PatternMatchingMatchPatternToSubsituteBadArg,
    Values(
        std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1;
                    GArray<int> in2;
                    GScalar in3;
                    GMat out1 = cv::gapi::addC(in1, in3);
                    GArray<int> out2 = GDummyArrayKernel::on(in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2.strip()));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2.strip()));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in1;
                    GArray<int> in2;
                    GMat in3;
                    GMat out1 = cv::gapi::add(in1, in3);
                    GArray<int> out2 = GDummyArrayKernel::on(in2);
                    matching_test::initGModel(*sg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2.strip()));
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2.strip()));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            }),
        std::make_tuple([] () {
                std::unique_ptr<ade::Graph> pg(new ade::Graph);
                std::vector<ade::NodeHandle> p_ins, p_outs;
                {
                    GMat in1;
                    GArray<int> in2;
                    GScalar in3;
                    GMat out1 = cv::gapi::addC(in1, in3);
                    GArray<int> out2 = GDummyArrayKernel::on(in2);
                    matching_test::initGModel(*pg, cv::GIn(in1, in2, in3), cv::GOut(out1, out2));
                    cv::gimpl::GModel::Graph m(*pg);
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in1));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in2.strip()));
                    p_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in3));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out1));
                    p_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out2.strip()));
                }
                return std::make_tuple(std::move(pg), std::move(p_ins), std::move(p_outs));
            }, [] () {
                std::unique_ptr<ade::Graph> sg(new ade::Graph);
                std::vector<ade::NodeHandle> s_ins, s_outs;
                {
                    GMat in;
                    GMat out = cv::gapi::bitwise_not(in);
                    matching_test::initGModel(*sg, cv::GIn(in), cv::GOut(out));
                    cv::gimpl::GModel::Graph m(*sg);
                    s_ins.emplace_back(cv::gimpl::GModel::dataNodeOf(m, in));
                    s_outs.emplace_back(cv::gimpl::GModel::dataNodeOf(m, out));
                }
                return std::make_tuple(std::move(sg), std::move(s_ins), std::move(s_outs));
            })));

// --------------------------------------------------------------------------------------
// Accuracy integration tests (GComputation-level)

namespace {
// custom "listener" to check what kernels are called within the test
struct KernelListener { std::map<std::string, size_t> counts; };
KernelListener& getListener() {
    static KernelListener l;
    return l;
}

using CompCreator = std::function<cv::GComputation()>;
using CompileArgsCreator = std::function<cv::GCompileArgs()>;
using Verifier = std::function<void(KernelListener)>;
}  // anonymous namespace

struct PatternMatchingIntegration :
    TestWithParam<std::tuple<CompCreator, Verifier, cv::GCompileArgs, cv::GCompileArgs>> {};
TEST_P(PatternMatchingIntegration, TestApplyTransformation) {
    cv::Size in_sz(640, 480);
    cv::Mat y(in_sz, CV_8UC1), uv(cv::Size(in_sz.width / 2, in_sz.height / 2), CV_8UC2);
    cv::randu(y, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::randu(y, cv::Scalar::all(100), cv::Scalar::all(200));
    cv::Mat orig_output, transformed_output;

    auto orig_args = std::get<2>(GetParam());  // returns compile args for original graph
    auto transform_args = std::get<3>(GetParam());  // returns compile args with transformations

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    {
        // Run original graph
        auto mainC = std::get<0>(GetParam())();
        mainC.apply(cv::gin(y, uv), cv::gout(orig_output), std::move(orig_args));
    }

    // Generate transformed graph (passing transformations via compile args)
    auto mainC = std::get<0>(GetParam())();  // get new copy with new Priv
    mainC.apply(cv::gin(y, uv), cv::gout(transformed_output), std::move(transform_args));

    // Compare
    ASSERT_TRUE(AbsExact()(orig_output, transformed_output));

    // Custom verification via listener
    auto verify = std::get<1>(GetParam());
    verify(listener);
}

// custom NV12
G_TYPED_KERNEL(MyNV12toBGR, <GMat(GMat, GMat)>, "test.my_nv12_to_bgr") {
    static GMatDesc outMeta(GMatDesc in_y, GMatDesc in_uv) {
        return cv::gapi::imgproc::GNV12toBGR::outMeta(in_y, in_uv);
    }
};
GAPI_OCV_KERNEL(MyNV12toBGRImpl, MyNV12toBGR)
{
    static void run(const cv::Mat& in_y, const cv::Mat& in_uv, cv::Mat &out)
    {
        getListener().counts[MyNV12toBGR::id()]++;
        cv::cvtColorTwoPlane(in_y, in_uv, out, cv::COLOR_YUV2BGR_NV12);
    }
};
G_TYPED_KERNEL(MyPlanarResize, <GMatP(GMatP, Size, int)>, "test.my_planar_resize") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int interp) {
        return cv::gapi::core::GResizeP::outMeta(in, sz, interp);
    }
};
GAPI_OCV_KERNEL(MyPlanarResizeImpl, MyPlanarResize) {
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat &out)
    {
        getListener().counts[MyPlanarResize::id()]++;
        int inH = in.rows / 3;
        int inW = in.cols;
        int outH = out.rows / 3;
        int outW = out.cols;
        for (int i = 0; i < 3; i++) {
            auto in_plane = in(cv::Rect(0, i*inH, inW, inH));
            auto out_plane = out(cv::Rect(0, i*outH, outW, outH));
            cv::resize(in_plane, out_plane, out_sz, 0, 0, interp);
        }
    }
};
G_TYPED_KERNEL(MyInterleavedResize, <GMat(GMat, Size, int)>, "test.my_planar_resize") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int interp) {
        return cv::gapi::core::GResize::outMeta(in, sz, 0.0, 0.0, interp);
    }
};
GAPI_OCV_KERNEL(MyInterleavedResizeImpl, MyInterleavedResize) {
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat &out)
    {
        getListener().counts[MyInterleavedResize::id()]++;
        cv::resize(in, out, out_sz, 0.0, 0.0, interp);
    }
};
GAPI_OCV_KERNEL(MyToNCHWImpl, GToNCHW) {
    static void run(const cv::Mat& in, cv::Mat& out)
    {
        getListener().counts[GToNCHW::id()]++;
        auto sz = in.size();
        auto w = sz.width;
        auto h = sz.height;
        cv::Mat ins[3] = {};
        cv::split(in, ins);
        for (int i = 0; i < 3; i++) {
            auto in_plane = ins[i];
            auto out_plane = out(cv::Rect(0, i*h, w, h));
            in_plane.copyTo(out_plane);
        }
    }
};

GAPI_TRANSFORM(NV12Transform, <cv::GMat(cv::GMat, cv::GMat)>, "test.nv12_transform")
{
    static cv::GMat pattern(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = MyNV12toBGR::on(y, uv);
        return out;
    }

    static cv::GMat substitute(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = cv::gapi::NV12toBGR(y, uv);
        return out;
    }
};
GAPI_TRANSFORM(NV12TransformReversed, <cv::GMat(cv::GMat, cv::GMat)>, "test.nv12_transform_rev")
{
    static cv::GMat pattern(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = cv::gapi::NV12toBGR(y, uv);
        return out;
    }

    static cv::GMat substitute(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = MyNV12toBGR::on(y, uv);
        return out;
    }
};
GAPI_TRANSFORM(ResizeTransform, <cv::GMat(cv::GMat)>, "test.resize_transform")
{
    static cv::GMat pattern(const cv::GMat& in)
    {
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
            cv::Size(100, 100), 0, 0, cv::INTER_AREA);
        return cv::gapi::merge3(resize(b), resize(g), resize(r));
    }

    static cv::GMat substitute(const cv::GMat& in)
    {
        return MyInterleavedResize::on(in, cv::Size(100, 100), cv::INTER_AREA);
    }
};
GAPI_TRANSFORM(ChainTransform1, <GMatP(GMat)>, "Resize + toNCHW -> toNCHW + PlanarResize")
{
    static GMatP pattern(const cv::GMat& in)
    {
        return GToNCHW::on(cv::gapi::resize(in, cv::Size(60, 60)));
    }

    static GMatP substitute(const cv::GMat& in)
    {
        return MyPlanarResize::on(GToNCHW::on(in), cv::Size(60, 60), cv::INTER_LINEAR);
    }
};
GAPI_TRANSFORM(ChainTransform2, <GMatP(GMat, GMat)>, "NV12toBGR + toNCHW -> NV12toBGRp")
{
    static GMatP pattern(const GMat& y, const GMat& uv)
    {
        return GToNCHW::on(MyNV12toBGR::on(y, uv));
    }

    static GMatP substitute(const GMat& y, const GMat& uv)
    {
        return cv::gapi::NV12toBGRp(y, uv);
    }
};

INSTANTIATE_TEST_CASE_P(Everything, PatternMatchingIntegration,
    Values(
        std::make_tuple(
            [] () {
                GMat in1, in2;
                GMat bgr = MyNV12toBGR::on(in1, in2);
                GMat out = cv::gapi::resize(bgr, cv::Size(100, 100));
                return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
            },
            [] (const KernelListener& l) {
                ASSERT_EQ(1u, l.counts.size());
                // called in original graph:
                ASSERT_NE(l.counts.cend(), l.counts.find(MyNV12toBGR::id()));
                ASSERT_EQ(1u, l.counts.at(MyNV12toBGR::id()));
            },
            cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl>()),
            cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, NV12Transform>())),
        std::make_tuple(
            [] () {
                GMat in1, in2;
                GMat bgr = cv::gapi::NV12toBGR(in1, in2);
                GMat b, g, r;
                std::tie(b, g, r) = cv::gapi::split3(bgr);
                const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
                    cv::Size(100, 100), 0, 0, cv::INTER_AREA);
                GMat tmp1 = cv::gapi::resize(bgr, cv::Size(90, 90));
                GMat tmp2 = cv::gapi::bitwise_not(cv::gapi::merge3(resize(b), resize(g), resize(r)));
                GMat out = cv::gapi::add(
                    cv::gapi::resize(cv::gapi::addC(tmp1, GScalar(10.0)), cv::Size(100, 100)),
                    tmp2);
                return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
            },
            [] (const KernelListener& l) {
                ASSERT_EQ(2u, l.counts.size());
                // called in transformed graph:
                ASSERT_NE(l.counts.cend(), l.counts.find(MyNV12toBGR::id()));
                ASSERT_EQ(1u, l.counts.at(MyNV12toBGR::id()));
                ASSERT_NE(l.counts.cend(), l.counts.find(MyInterleavedResize::id()));
                ASSERT_EQ(1u, l.counts.at(MyInterleavedResize::id()));
            },
            cv::compile_args(),
            cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, MyInterleavedResizeImpl,
                                               ResizeTransform, NV12TransformReversed>()))
    ));

INSTANTIATE_TEST_CASE_P(BasicE2E, PatternMatchingIntegration,
    Combine(
        Values([] () {
                GMat in1, in2;
                GMat bgr = MyNV12toBGR::on(in1, in2);
                GMat resized = cv::gapi::resize(bgr, cv::Size(60, 60));
                GMatP out = GToNCHW::on(resized);
                return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
        }),
        Values([] (const KernelListener& l) {
                ASSERT_EQ(3u, l.counts.size());
                // called in original graph:
                ASSERT_NE(l.counts.cend(), l.counts.find(MyNV12toBGR::id()));
                ASSERT_NE(l.counts.cend(), l.counts.find(GToNCHW::id()));
                ASSERT_EQ(1u, l.counts.at(MyNV12toBGR::id()));
                ASSERT_EQ(1u, l.counts.at(GToNCHW::id()));
                // called in transformed graph:
                ASSERT_NE(l.counts.cend(), l.counts.find(MyPlanarResize::id()));
                ASSERT_EQ(1u, l.counts.at(MyPlanarResize::id()));
            }),
        Values(cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, MyToNCHWImpl>())),
        Values(  // NB: must work regardless of transformations order
            cv::compile_args(
                cv::gapi::kernels<MyPlanarResizeImpl, ChainTransform1, ChainTransform2>()),
            cv::compile_args(
                cv::gapi::kernels<ChainTransform2, ChainTransform1, MyPlanarResizeImpl>()))));

// --------------------------------------------------------------------------------------
// Bad arg integration tests (GCompiler-level)

struct PatternMatchingIntegrationEndlessLoops :
    TestWithParam<std::tuple<CompCreator, cv::GCompileArgs>> {};
TEST_P(PatternMatchingIntegrationEndlessLoops, TestTransformationThrows) {
    cv::Size in_sz(640, 480);
    cv::Mat input(in_sz, CV_8UC3);
    cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::Mat orig_output, transformed_output;

    auto mainC = std::get<0>(GetParam())();  // get new copy with new Priv
    auto args = std::get<1>(GetParam());
    cv::gimpl::GCompiler compiler(mainC, cv::descr_of(cv::gin(input)), std::move(args));

    EXPECT_THROW(compiler.compile(), std::exception);  //  should throw here
}

GAPI_TRANSFORM(EndlessLoopTransform1, <cv::GMat(cv::GMat)>, "test.endless_loop_transform1")
{
    static cv::GMat pattern(const cv::GMat& in)
    {
        return cv::gapi::resize(in, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
    }

    static cv::GMat substitute(const cv::GMat& in)
    {
        cv::GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        auto resize = std::bind(&cv::gapi::resize,
            std::placeholders::_1, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
        cv::GMat out = cv::gapi::merge3(resize(b), resize(g), resize(r));
        return out;
    }
};

// FIXME: add more tests for different kinds of loops (when the logic is there)
INSTANTIATE_TEST_CASE_P(All, PatternMatchingIntegrationEndlessLoops,
    Values(std::make_tuple(
                [] () {
                    GMat in;
                    GMat tmp = cv::gapi::resize(in, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
                    GMat out = cv::gapi::bitwise_not(tmp);
                    return cv::GComputation(cv::GIn(in), cv::GOut(out));
                },
                cv::compile_args(cv::gapi::kernels<EndlessLoopTransform1>())
            )));

} // namespace opencv_test
