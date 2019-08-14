// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILER_HPP
#define OPENCV_GAPI_GCOMPILER_HPP


#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/gcomputation.hpp>

#include <ade/execution_engine/execution_engine.hpp>

#include "gmodel.hpp"

namespace cv { namespace gimpl {

// FIXME: exported for internal tests only!
class GAPI_EXPORTS GCompiler
{
    const GComputation&      m_c;
    const GMetaArgs          m_metas;
    GCompileArgs             m_args;
    ade::ExecutionEngine     m_e;

    cv::gapi::GKernelPackage m_all_kernels;
    cv::gapi::GNetPackage    m_all_networks;

    void validateInputMeta();
    void validateOutProtoArgs();

public:
    explicit GCompiler(const GComputation &c,
                             GMetaArgs    &&metas,
                             GCompileArgs &&args);

    // The method which does everything...
    GCompiled compile();

    // But is actually composed of this:
    using GPtr = std::unique_ptr<ade::Graph>;
    GPtr       generateGraph();               // Unroll GComputation into a GModel
    void       runPasses(ade::Graph &g);      // Apply all G-API passes on a GModel
    void       compileIslands(ade::Graph &g); // Instantiate GIslandExecutables in GIslandModel
    GCompiled  produceCompiled(GPtr &&pg);    // Produce GCompiled from processed GModel

    // FIXME: can't use optional -> ade::Graph/GModel::Graph not copy-able
    bool transform(GModel::Graph& main, const GModel::Graph& pattern,
        const GModel::Graph& substitute);

    // FIXME: main && maing - the same thing
    bool transform(ade::Graph& main, GModel::Graph& maing,
        const GModel::Graph& pattern,
        const GModel::Graph& substitute,
        const cv::GProtoArgs& substitute_ins, const cv::GProtoArgs& substitute_outs,
        const cv::GMetaArgs& substitute_metas);
};

}}

#endif // OPENCV_GAPI_GCOMPILER_HPP
