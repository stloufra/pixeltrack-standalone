// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_BLAS_TRAITS_TRAITS_HPP
#define CUBLASDX_BLAS_TRAITS_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "../detail/blas_description_fd.hpp"

#include "../operators.hpp"
#include "../types.hpp"

#include "detail/description_traits.hpp"

#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

namespace cublasdx {
    template<class Description>
    using precision_of = commondx::precision_of<operator_type, Description, detail::default_blas_precision_operator>;

    template<class Description>
    using precision_of_t = typename precision_of<Description>::type;

    template<class Description>
    using type_of = commondx::data_type_of<operator_type, Description, detail::default_blas_type_operator>;

    template<class Description>
    inline constexpr type type_of_v = type_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<operator_type, Description>;
    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    template<class Description>
    struct size_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

    public:
        using value_type                = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int m = detail::get_t<operator_type::size, Description>::m;
        static constexpr unsigned int n = detail::get_t<operator_type::size, Description>::n;
        static constexpr unsigned int k = detail::get_t<operator_type::size, Description>::k;

        static constexpr value_type value = value_type {m, n, k};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> size_of<Description>::value;
    template<class Description>
    constexpr unsigned int size_of<Description>::m;
    template<class Description>
    constexpr unsigned int size_of<Description>::n;
    template<class Description>
    constexpr unsigned int size_of<Description>::k;

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> size_of_v = size_of<Description>::value;

    template<class Description>
    struct function_of {
        using value_type                  = function;
        static constexpr value_type value = detail::get_t<operator_type::function, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr function function_of<Description>::value;

    template<class Description>
    inline constexpr function function_of_v = function_of<Description>::value;

    template<class Description>
    struct alignment_of {
        using value_type                        = unsigned int;
        static constexpr value_type a_alignment = detail::get_t<operator_type::alignment, Description>::a_alignment;
        static constexpr value_type b_alignment = detail::get_t<operator_type::alignment, Description>::b_alignment;
        static constexpr value_type c_alignment = detail::get_t<operator_type::alignment, Description>::c_alignment;
    };

    template<class Description>
    constexpr unsigned int alignment_of<Description>::a_alignment;
    template<class Description>
    constexpr unsigned int alignment_of<Description>::b_alignment;
    template<class Description>
    constexpr unsigned int alignment_of<Description>::c_alignment;

    namespace detail {
        template<unsigned int M, unsigned int N, unsigned int K, class TransposeMode>
        struct default_leading_dimension {
            using type = LeadingDimension<((TransposeMode::a_transpose_mode == transpose_mode::non_transposed) ? M : K),
                                          ((TransposeMode::b_transpose_mode == transpose_mode::non_transposed) ? K : N),
                                          M>;
        };

        template<unsigned int M, unsigned int N, unsigned int K>
        struct default_leading_dimension<M, N, K, void> {
            using type = LeadingDimension<M, K, M>;
        };

        template<unsigned int M, unsigned int N, unsigned int K, class TransposeMode>
        using default_leading_dimension_t = typename default_leading_dimension<M, N, K, TransposeMode>::type;
    } // namespace detail

    template<class Description>
    struct leading_dimension_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static constexpr bool has_ld   = detail::has_operator<operator_type::ld, Description>::value;
        static_assert(has_size || has_ld, "Description does not have size nor leading dimensions defined");

        static constexpr auto m = size_of<Description>::m;
        static constexpr auto n = size_of<Description>::n;
        static constexpr auto k = size_of<Description>::k;
        using default_ld =
            detail::default_leading_dimension_t<m, n, k, detail::get_t<operator_type::transpose_mode, Description>>;

    public:
        using value_type                = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a = detail::get_or_default_t<operator_type::ld, Description, default_ld>::a;
        static constexpr unsigned int b = detail::get_or_default_t<operator_type::ld, Description, default_ld>::b;
        static constexpr unsigned int c = detail::get_or_default_t<operator_type::ld, Description, default_ld>::c;

        static constexpr value_type value = value_type {a, b, c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> leading_dimension_of<Description>::value;
    template<class Description>
    constexpr unsigned int leading_dimension_of<Description>::a;
    template<class Description>
    constexpr unsigned int leading_dimension_of<Description>::b;
    template<class Description>
    constexpr unsigned int leading_dimension_of<Description>::c;

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> leading_dimension_of_v = leading_dimension_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_a = leading_dimension_of<Description>::a;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_b = leading_dimension_of<Description>::b;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_c = leading_dimension_of<Description>::c;

    // template<class Description>
    // struct fill_mode_of {
    //     using value_type                  = fill_mode;
    //     static constexpr value_type value = detail::get_t<operator_type::fill_mode, Description>::value;
    //     constexpr                   operator value_type() const noexcept { return value; }
    // };

    // template<class Description>
    // constexpr fill_mode fill_mode_of<Description>::value;

    // template<class Description>
    // inline constexpr fill_mode fill_mode_of_v = fill_mode_of<Description>::value;

    template<class Description>
    struct transpose_mode_of {
        using value_type = transpose_mode;
        static constexpr value_type a_transpose_mode =
            detail::get_t<operator_type::transpose_mode, Description>::a_transpose_mode;
        static constexpr value_type b_transpose_mode =
            detail::get_t<operator_type::transpose_mode, Description>::b_transpose_mode;
    };

    template<class Description>
    constexpr transpose_mode transpose_mode_of<Description>::a_transpose_mode;
    template<class Description>
    constexpr transpose_mode transpose_mode_of<Description>::b_transpose_mode;

    template<class Description>
    inline constexpr transpose_mode transpose_mode_of_a = transpose_mode_of<Description>::a_transpose_mode;
    template<class Description>
    inline constexpr transpose_mode transpose_mode_of_b = transpose_mode_of<Description>::b_transpose_mode;

    // template<class Description>
    // struct diagonal_of {
    //     using value_type                  = diagonal;
    //     static constexpr value_type value = detail::get_t<operator_type::diagonal, Description>::value;
    //     constexpr                   operator value_type() const noexcept { return value; }
    // };

    // template<class Description>
    // constexpr diagonal diagonal_of<Description>::value;

    // template<class Description>
    // inline constexpr diagonal diagonal_of_v = diagonal_of<Description>::value;

    // template<class Description>
    // struct side_of {
    //     using value_type                  = side;
    //     static constexpr value_type value = detail::get_t<operator_type::side, Description>::value;
    //     constexpr                   operator value_type() const noexcept { return value; }
    // };

    // template<class Description>
    // constexpr side side_of<Description>::value;

    // template<class Description>
    // inline constexpr side side_of_v = side_of<Description>::value;

    template<class Description>
    using is_blas = commondx::is_dx_expression<Description>;

    template<class Description>
    using is_blas_execution = commondx::is_dx_execution_expression<operator_type, Description>;

    template<class Description>
    using is_complete_blas = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    using is_complete_blas_execution =
        commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>;

    template<class Description>
    using extract_blas_description = commondx::extract_dx_description<detail::blas_description, Description, operator_type>;

    template<class Description>
    using extract_blas_description_t = typename extract_blas_description<Description>::type;

    namespace detail {
        // forward declaration
        template<class Precision,
                 cublasdx::type Type,
                 transpose_mode ATMode,
                 transpose_mode BTMode,
                 class Size,
                 class LD,
                 unsigned int Arch>
        struct is_supported_impl;

        template<class Size, class TransposeMode, class Precision, type Type, unsigned int Architecture>
        struct suggested_leading_dimension_of_impl {
            static constexpr auto m                = Size::m;
            static constexpr auto n                = Size::n;
            static constexpr auto k                = Size::k;
            static constexpr auto transpose_mode_a = TransposeMode::a_transpose_mode;
            static constexpr auto transpose_mode_b = TransposeMode::b_transpose_mode;
            using value_type = COMMONDX_STL_NAMESPACE::conditional_t<(Type == type::complex), complex<Precision>, Precision>;

            // Suggested leading dimension
            static constexpr bool m_padding_required =
                ((m * sizeof(value_type)) >= 128) && (((m * sizeof(value_type)) % 128) == 0);
            static constexpr bool n_padding_required =
                ((n * sizeof(value_type)) >= 128) && (((n * sizeof(value_type)) % 128) == 0);
            static constexpr bool k_padding_required =
                ((k * sizeof(value_type)) >= 128) && (((k * sizeof(value_type)) % 128) == 0);

            static constexpr unsigned int padded_m = m_padding_required ? (m + (16 / sizeof(value_type))) : m;
            static constexpr unsigned int padded_n = n_padding_required ? (n + (16 / sizeof(value_type))) : n;
            static constexpr unsigned int padded_k = k_padding_required ? (k + (16 / sizeof(value_type))) : k;

            static constexpr unsigned int best_lda = (transpose_mode_a == N) ? padded_m : padded_k;
            static constexpr unsigned int best_ldb = (transpose_mode_b == N) ? padded_k : padded_n;
            static constexpr unsigned int best_ldc = m;

            // Default leading dimensions
            using default_ld = default_leading_dimension_t<m, n, k, TransposeMode>;

            // Check if the operation is supported with LD(best_lda, best_ldb, best_ldc)
            static constexpr bool padding_supported = is_supported_impl<Precision,
                                                                        Type,
                                                                        transpose_mode_a,
                                                                        transpose_mode_b,
                                                                        Size,
                                                                        LeadingDimension<best_lda, best_ldb, best_ldc>,
                                                                        Architecture>::value;

            // Use default or best leading dimensions
            static constexpr unsigned int lda = padding_supported ? best_lda : default_ld::a;
            static constexpr unsigned int ldb = padding_supported ? best_ldb : default_ld::b;
            static constexpr unsigned int ldc = padding_supported ? best_ldc : default_ld::c;
        };
    } // namespace detail

    template<class Description, unsigned int Architecture>
    struct suggested_leading_dimension_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

        static constexpr bool has_transpose = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_transpose, "Description does not have transpose mode defined");

        using blas_precision_t            = precision_of_t<Description>;
        static constexpr auto blas_type_v = type_of_v<Description>;

        using suggested =
            detail::suggested_leading_dimension_of_impl<detail::get_t<operator_type::size, Description>,
                                                        detail::get_t<operator_type::transpose_mode, Description>,
                                                        blas_precision_t,
                                                        blas_type_v,
                                                        Architecture>;

    public:
        static constexpr unsigned int lda = suggested::lda;
        static constexpr unsigned int ldb = suggested::ldb;
        static constexpr unsigned int ldc = suggested::ldc;
        // Suggested leading dimensions of A, B matrices that can provide a good performance
        using type = LeadingDimension<lda, ldb, ldc>;
    };

    template<class Description, unsigned int Architecture>
    constexpr unsigned int suggested_leading_dimension_of<Description, Architecture>::lda;
    template<class Description, unsigned int Architecture>
    constexpr unsigned int suggested_leading_dimension_of<Description, Architecture>::ldb;
    template<class Description, unsigned int Architecture>
    constexpr unsigned int suggested_leading_dimension_of<Description, Architecture>::ldc;

    template<class Description, unsigned int Architecture>
    using suggested_leading_dimension_of_t = typename suggested_leading_dimension_of<Description, Architecture>::type;
} // namespace cublasdx

#endif // CUBLASDX_BLAS_TRAITS_TRAITS_HPP
