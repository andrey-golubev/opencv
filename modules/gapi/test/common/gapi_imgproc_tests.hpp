// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_HPP

#include <iostream>

#include "gapi_tests_common.hpp"

namespace opencv_test
{

struct Filter2DTest : public TestWithParamBase<compare_f,int,int>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, kernSize, borderType);
    USE_NORMAL_INIT(Filter2DTest);
};
struct BoxFilterTest : public TestWithParamBase<compare_f,int,int>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, filterSize, borderType);
    USE_NORMAL_INIT(BoxFilterTest);
};
struct SepFilterTest : public TestWithParamBase<compare_f,int>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, kernSize);
    USE_NORMAL_INIT(SepFilterTest);
};
struct BlurTest : public TestWithParamBase<compare_f,int,int>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, filterSize, borderType);
    USE_NORMAL_INIT(BlurTest);
};
struct GaussianBlurTest : public TestWithParamBase<compare_f,int>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, kernSize);
    USE_NORMAL_INIT(GaussianBlurTest);
};
struct MedianBlurTest : public TestWithParamBase<compare_f,int>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, kernSize);
    USE_NORMAL_INIT(MedianBlurTest);
};
struct ErodeTest : public TestWithParamBase<compare_f,int,int>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, kernSize, kernType);
    USE_NORMAL_INIT(ErodeTest);
};
struct Erode3x3Test : public TestWithParamBase<compare_f,int>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, numIters);
    USE_NORMAL_INIT(Erode3x3Test);
};
struct DilateTest : public TestWithParamBase<compare_f,int,int>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, kernSize, kernType);
    USE_NORMAL_INIT(DilateTest);
};
struct Dilate3x3Test : public TestWithParamBase<compare_f,int>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, numIters);
    USE_NORMAL_INIT(Dilate3x3Test);
};
struct SobelTest : public TestWithParamBase<compare_f,int,int,int>
{
    DEFINE_SPECIFIC_PARAMS_4(cmpF, kernSize, dx, dy);
    USE_NORMAL_INIT(SobelTest);
};
struct EqHistTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(EqHistTest);
};
struct CannyTest : public TestWithParamBase<compare_f,double,double,int,bool>
{
    DEFINE_SPECIFIC_PARAMS_5(cmpF, thrLow, thrUp, apSize, l2gr);
    USE_NORMAL_INIT(CannyTest);
};
struct RGB2GrayTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(RGB2GrayTest);
};
struct BGR2GrayTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(BGR2GrayTest);
};
struct RGB2YUVTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(RGB2YUVTest);
};
struct YUV2RGBTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(YUV2RGBTest);
};
struct RGB2LabTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(RGB2LabTest);
};
struct BGR2LUVTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(BGR2LUVTest);
};
struct LUV2BGRTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(LUV2BGRTest);
};
struct BGR2YUVTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(BGR2YUVTest);
};
struct YUV2BGRTest : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_NORMAL_INIT(YUV2BGRTest);
};
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_HPP
