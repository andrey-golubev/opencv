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
}  // anonymous namespace
}  // namespace matching_test

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

} // namespace opencv_test
