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

namespace detail
{
template<typename _TupleIn, typename _TupleOut, size_t RFirst, size_t RLast, size_t IndexOut,
    bool stop = false>
struct CopyUtils
{
    inline static void copy(const _TupleIn& in, _TupleOut& out)
    {
        static_assert(std::is_same<
            typename std::tuple_element<RFirst, _TupleIn>::type,
            typename std::tuple_element<IndexOut, _TupleOut>::type
            >::value, "types(in) != types(out) within the range of copying");
        std::get<IndexOut>(out) = std::get<RFirst>(in);
        CopyUtils<_TupleIn, _TupleOut, RFirst+1, RLast, IndexOut+1,
            (IndexOut+1 >= std::tuple_size<_TupleOut>::value || RFirst+1 >= RLast)>::copy(in, out);
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

// copy range of values from `in` tuple to `out` tuple
template<size_t RFirst, size_t RLast, size_t Index = 0, typename _TupleIn, typename _TupleOut>
inline static void copyTupleRange(const _TupleIn& in,  _TupleOut& out)
{
    static_assert(Index < std::tuple_size<_TupleOut>::value, "out tuple index out of bounds");
    static_assert(RFirst < RLast, "invalid length of range");
    static_assert((RLast - RFirst) <= std::tuple_size<_TupleIn>::value,
        "range is bigger than input tuple");
    static_assert((RLast - RFirst) <= std::tuple_size<_TupleOut>::value,
        "range is bigger than output tuple");
    CopyUtils<_TupleIn, _TupleOut, RFirst, RLast, Index>::copy(in, out);
}
} // namespace detail

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
    __DEFINE_PARAMS_IMPL3(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_4(...) \
    __DEFINE_PARAMS_IMPL4(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_5(...) \
    __DEFINE_PARAMS_IMPL5(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)

#define DEFINE_SPECIFIC_PARAMS_6(...) \
    __DEFINE_PARAMS_IMPL6(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__)
} // namespace opencv_test

#endif //OPENCV_GAPI_TESTS_HELPERS_HPP
