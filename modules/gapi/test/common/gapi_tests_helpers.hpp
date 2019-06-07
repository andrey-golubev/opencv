// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_GAPI_TESTS_HELPERS_HPP
#define OPENCV_GAPI_TESTS_HELPERS_HPP

#include <tuple>
#include <limits>

namespace opencv_test
{

// TODO: can we use -1 instead?
// TODO: better name
// specifies that in_type == out_type in matrices initialization
enum {
    ALIGNED_TYPE = std::numeric_limits<int>::max()
};

// implementation of recursive in-class declaration and initialization of member variables
#define __DEFINE_PARAMS_IMPL1(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params);

#define __DEFINE_PARAMS_IMPL2(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    __DEFINE_PARAMS_IMPL1(params_type, params, index+1, __VA_ARGS__)

#define __DEFINE_PARAMS_IMPL3(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    __DEFINE_PARAMS_IMPL2(params_type, params, index+1, __VA_ARGS__)

#define __DEFINE_PARAMS_IMPL4(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    __DEFINE_PARAMS_IMPL3(params_type, params, index+1, __VA_ARGS__)

#define __DEFINE_PARAMS_IMPL5(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    __DEFINE_PARAMS_IMPL4(params_type, params, index+1, __VA_ARGS__)

#define __DEFINE_PARAMS_IMPL6(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    __DEFINE_PARAMS_IMPL5(params_type, params, index+1, __VA_ARGS__)

// user interface
#define DEFINE_SPECIFIC_PARAMS_1(...) \
    __DEFINE_PARAMS_IMPL1(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_2(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_3(...) \
    __DEFINE_PARAMS_IMPL3(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_4(...) \
    __DEFINE_PARAMS_IMPL4(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_5(...) \
    __DEFINE_PARAMS_IMPL5(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_6(...) \
    __DEFINE_PARAMS_IMPL6(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)
} // namespace opencv_test

#endif //OPENCV_GAPI_TESTS_HELPERS_HPP
