// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_GAPI_TESTS_HELPERS_HPP
#define OPENCV_GAPI_TESTS_HELPERS_HPP

#include <tuple>

namespace opencv_test
{
namespace tuple_extensions
{
template<typename _TupleIn, typename _TupleOut, size_t RFirst, size_t RLast, size_t IndexOut,
    bool stop = false>
struct CopyUtils
{
    inline static void copy(const _TupleIn& in, _TupleOut& out)
    {
        std::get<IndexOut>(out) = std::get<RFirst>(in);
        CopyUtils<_TupleIn, _TupleOut, RFirst+1, RLast, IndexOut+1,
            (std::tuple_size<_TupleOut>::value <= IndexOut+1 || RFirst+1 >= RLast)>::copy(in, out);
    }
};

template<typename _TupleIn, typename _TupleOut, size_t RFirst, size_t RLast, size_t IndexOut>
struct CopyUtils<_TupleIn, _TupleOut, RFirst, RLast, IndexOut, true>
{
    inline static void copy(const _TupleIn&, _TupleOut&)
    {
        // base case
    }
};

// copy range of values from src tuple to dst tuple
template<size_t RFirst, size_t RLast, size_t Index = 0, typename _TupleIn, typename _TupleOut>
inline static void copyFromRange(const _TupleIn& src,  _TupleOut& dst)
{
    static_assert(Index < std::tuple_size<_TupleOut>::value, "Dst tuple index out of bounds");
    static_assert(RFirst < RLast, "Invalid length of range");
    static_assert((RLast - RFirst) <= std::tuple_size<_TupleOut>::value,
        "Range is bigger than dst tuple");
    CopyUtils< _TupleIn, _TupleOut, RFirst, RLast, Index, (RFirst >= RLast)>::copy(src, dst);
}
} // namespace tuple_extensions

// in-class definition and initialization of member variables (test parameters)
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

// interface to define in-class member variables (test parameters)
#define DEFINE_SPECIFIC_PARAMS_1(...) \
    __DEFINE_PARAMS_IMPL1(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_2(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_3(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_4(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_5(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_6(...) \
    __DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)
} // namespace opencv_test

#endif //OPENCV_GAPI_TESTS_HELPERS_HPP
