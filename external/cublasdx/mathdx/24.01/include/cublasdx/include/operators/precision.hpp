// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_PRECISION_HPP
#define CUBLASDX_OPERATORS_PRECISION_HPP

#include "commondx/operators/precision.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    template<class T>
    struct Precision: public commondx::PrecisionBase<T, __half, float, double> {
    };

    namespace detail {
        using default_blas_precision_operator = Precision<float>;
    } // namespace detail
} // namespace cublasdx

namespace commondx::detail {
    template<class T>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::precision, cublasdx::Precision<T>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<class T>
    struct get_operator_type<cublasdx::Precision<T>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::precision;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_PRECISION_HPP
