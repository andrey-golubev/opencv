// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_INL_HPP

#include "opencv2/gapi/imgproc.hpp"
#include "gapi_imgproc_tests.hpp"

namespace opencv_test
{
TEST_P(Filter2DTest, AccuracyTest)
{
    cv::Point anchor = {-1, -1};
    double delta = 0;

    cv::Mat kernel = cv::Mat(kernSize, kernSize, CV_32FC1);
    cv::Scalar kernMean = cv::Scalar(1.0);
    cv::Scalar kernStddev = cv::Scalar(2.0/3);
    randn(kernel, kernMean, kernStddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::filter2D(in_mat1, out_mat_ocv, dtype, kernel, anchor, delta, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BoxFilterTest, AccuracyTest)
{
    cv::Point anchor = {-1, -1};
    bool normalize = true;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::boxFilter(in, dtype, cv::Size(filterSize, filterSize), anchor, normalize,
        borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::boxFilter(in_mat1, out_mat_ocv, dtype, cv::Size(filterSize, filterSize), anchor,
            normalize, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(SepFilterTest, AccuracyTest)
{
    cv::Mat kernelX(kernSize, 1, CV_32F);
    cv::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);

    cv::Point anchor = cv::Point(-1, -1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, cv::Scalar() );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::sepFilter2D(in_mat1, out_mat_ocv, dtype, kernelX, kernelY );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BlurTest, AccuracyTest)
{
    cv::Point anchor = {-1, -1};

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::blur(in, cv::Size(filterSize, filterSize), anchor, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::blur(in_mat1, out_mat_ocv, cv::Size(filterSize, filterSize), anchor, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(GaussianBlurTest, AccuracyTest)
{
    cv::Size kSize = cv::Size(kernSize, kernSize);
    double sigmaX = rand();

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::gaussianBlur(in, kSize, sigmaX);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(MedianBlurTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::medianBlur(in, kernSize);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(ErodeTest, AccuracyTest)
{
    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(Erode3x3Test, AccuracyTest)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel, cv::Point(-1, -1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(DilateTest, AccuracyTest)
{
    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(Dilate3x3Test, AccuracyTest)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel, cv::Point(-1,-1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(SobelTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Sobel(in, dtype, dx, dy, kernSize );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(EqHistTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::equalizeHist(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::equalizeHist(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(CannyTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(YUV2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2LabTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Lab(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2Lab);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2LUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2LUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2Luv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(LUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_Luv2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(YUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
