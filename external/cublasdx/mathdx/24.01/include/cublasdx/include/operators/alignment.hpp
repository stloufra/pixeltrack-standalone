// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_ALIGNMENT_HPP
#define CUBLASDX_OPERATORS_ALIGNMENT_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    struct Alignment: commondx::detail::operator_expression {
        static constexpr unsigned int a_alignment = AAlignment;
        static constexpr unsigned int b_alignment = BAlignment;
        static constexpr unsigned int c_alignment = CAlignment;
    };

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::a_alignment;

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::b_alignment;

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::c_alignment;
} // namespace cublasdx

namespace commondx::detail {
    template<unsigned int A, unsigned int B, unsigned int C>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::alignment, cublasdx::Alignment<A, B, C>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int A, unsigned int B, unsigned int C>
    struct get_operator_type<cublasdx::Alignment<A, B, C>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::alignment;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_ALIGNMENT_HPP
