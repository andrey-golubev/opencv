// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "precomp.hpp"

#include <ade/util/zip_range.hpp>
#include <ade/graph.hpp>

#include "api/gcomputation_priv.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/passes/passes.hpp"
#include "compiler/passes/pattern_matching.hpp"

namespace cv { namespace gimpl { namespace passes {
namespace
{
std::unique_ptr<ade::Graph> makeGraph(const cv::GComputation& c) {
    // Generate ADE graph from expression-based computation
    std::unique_ptr<ade::Graph> pG(new ade::Graph);
    ade::Graph& g = *pG;

    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    cv::gimpl::GModelBuilder builder(g);
    auto proto_slots = builder.put(c.priv().m_ins, c.priv().m_outs);

    // Store Computation's protocol in metadata
    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    gm.metadata().set(p);

    return pG;
}

bool transform(ade::Graph& main, const std::unique_ptr<ade::Graph>& patternG,
    const cv::GComputation& substitute) {
    GModel::Graph gm(main);

    // Note: if there are multiple matches, p must be applied several times (outside-scope work)
    // 1. fina pattern in graph
    auto match1 = findMatches(*patternG, gm);
    if (!match1.ok()) {
        return false;
    }

    // 2. build substitute graph inside the main graph
    cv::gimpl::GModelBuilder builder(main);
    const auto& proto_slots = builder.put(substitute.priv().m_ins, substitute.priv().m_outs);
    Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;

    // 3. match pattern ins/outs to substitute ins/outs
    auto match2 = matchPatternToSubstitute(*patternG, main,
        GModel::Graph(*patternG).metadata().get<Protocol>(), p);

    // FIXME: from the state perspective it's better to validate matchings prior to applying the
    //        transformations: graph can already be (partially) transformed
    GAPI_Assert(match2.partialOk());

    // 4. make substitution
    performSubstitution(gm, match1, match2);
    return true;
}
}  // anonymous namespace

void checkTransformations(ade::passes::PassContext&,  // FIXME: context is unused here
    const gapi::GKernelPackage& transformations,
    std::vector<std::unique_ptr<ade::Graph>>& patterns)
{
    const auto& transforms = transformations.get_transformations();
    const auto size = transforms.size();
    if (0 == size) return;
    GAPI_Assert(0 == patterns.size() || size == patterns.size());
    patterns.resize(size);
    std::vector<std::unique_ptr<ade::Graph>> substitutes(size);

    // FIXME: verify other types of endless loops

    // verify there are no patterns in substitutes
    for (size_t i = 0; i < size; ++i) {
        auto& p = patterns[i];
        if (p == nullptr) {
            p = makeGraph(transforms[i].pattern());
        }
        for (size_t j = i; j < size; ++j) {
            auto& s = substitutes[j];
            if (s == nullptr) {
                s = makeGraph(transforms[j].substitute());
            }

            auto matchInSubstitute = findMatches(*p, *s);
            if (!matchInSubstitute.empty()) {
                throw std::runtime_error("Error: pattern detected inside substitute");
            }
        }
    }
}

void applyTransformations(ade::passes::PassContext& ctx,
    const gapi::GKernelPackage& transformations,
    const std::vector<std::unique_ptr<ade::Graph>>& patterns)
{
    const auto& transforms = transformations.get_transformations();
    const auto size = transforms.size();
    if (0 == size) return;
    // Note: patterns are already generated at this point
    GAPI_Assert(patterns.size() == transforms.size());

    // transform as long as it is possible
    // Note: check_transformations pass must handle endless loops and such
    bool continueTransforming = true;
    while (continueTransforming)
    {
        continueTransforming = false;

        // iterate through every transformation and try to transform graph parts
        for (auto it : ade::util::zip(ade::util::toRange(transforms), ade::util::toRange(patterns)))
        {
            const auto& t = std::get<0>(it);
            auto& p = std::get<1>(it);  // Note: using pre-created graphs
            GAPI_Assert(nullptr != p);

            // Note: applying the same substitution as long as possible
            bool transformationApplied = true;
            while (transformationApplied)
            {
                transformationApplied = transform(ctx.graph, p, t.substitute());

                // if at least one transformation happend, it is possible that other transforms will
                // be applied in the next "round"
                continueTransforming |= transformationApplied;
            }
        }
    }
}
}  // namespace passes
}  // namespace gimpl
}  // namespace cv
