// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
#define CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP

#include "commondx/detail/stl/type_traits.hpp"

#include "commondx/traits/detail/get.hpp"
#include "commondx/detail/expressions.hpp"

#include "../operators.hpp"
#include "../traits/detail/description_traits.hpp"

#include "blas_checks.hpp"

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace cublasdx {
    namespace detail {
        constexpr unsigned int calculate_matrix_size(unsigned int ld, unsigned int x, unsigned int y, transpose_mode tmode) {
            return ld * ((tmode == N) ? y : x);
        }

        template<class... Operators>
        class blas_operator_wrapper: public commondx::detail::description_expression { };

        template<class... Operators>
        class blas_description: public commondx::detail::description_expression
        {
            using description_type = blas_operator_wrapper<Operators...>;

        protected:
            /// ---- Traits

            // Size
            // * Default value: NONE
            // * If there is no size, then dummy size is (8, 8, 8). This is required value for M, N sized don't break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_size           = has_operator<operator_type::size, description_type>::value;
            using dummy_default_blas_size            = Size<8, 8, 8>;
            using this_blas_size                     = get_or_default_t<operator_type::size, description_type, dummy_default_blas_size>;
            static constexpr auto this_blas_size_m_v = this_blas_size::m;
            static constexpr auto this_blas_size_n_v = this_blas_size::n;
            static constexpr auto this_blas_size_k_v = this_blas_size::k;

            // Type (real, complex)
            // * Default value: real
            using this_blas_type                   = get_or_default_t<operator_type::type, description_type, default_blas_type_operator>;
            static constexpr auto this_blas_type_v = this_blas_type::value;

            // Function
            // * Default value: NONE
            // * Dummy value: MM
            static constexpr bool has_function = has_operator<operator_type::function, description_type>::value;
            using dummy_default_blas_function  = Function<function::MM>;
            using this_blas_function           = get_or_default_t<operator_type::function, description_type, dummy_default_blas_function>;
            static constexpr auto this_blas_function_v = this_blas_function::value;

            // Precision
            // * Default: float
            using this_blas_precision   = get_or_default_t<operator_type::precision, description_type, default_blas_precision_operator>;
            using this_blas_precision_t = typename this_blas_precision::type;

            // SM
            // * Default value: NONE
            // * Dummy value: 700
            static constexpr bool has_sm         = has_operator<operator_type::sm, description_type>::value;
            using dummy_default_blas_sm          = SM<700>;
            using this_blas_sm                   = get_or_default_t<operator_type::sm, description_type, dummy_default_blas_sm>;
            static constexpr auto this_blas_sm_v = this_blas_sm::value;

            // TransposeMode
            // * Default value: NONE
            // * Dummy value: N, N
            static constexpr bool has_transpose_mode = has_operator<operator_type::transpose_mode, description_type>::value;
            using dummy_default_blas_transpose_mode  = TransposeMode<N, N>;
            using this_blas_transpose_mode =
                get_or_default_t<operator_type::transpose_mode, description_type, dummy_default_blas_transpose_mode>;
            static constexpr auto this_blas_transpose_mode_a = this_blas_transpose_mode::a_transpose_mode;
            static constexpr auto this_blas_transpose_mode_b = this_blas_transpose_mode::b_transpose_mode;

            // // FillMode
            // // * Default value: NONE
            // // * Dummy value: fill_mode::lower
            // static constexpr bool has_fill_mode = has_operator<operator_type::fill_mode, description_type>::value;
            // using dummy_default_blas_fill_mode  = FillMode<fill_mode::lower>;
            // using this_blas_fill_mode = get_or_default_t<operator_type::fill_mode, description_type, dummy_default_blas_fill_mode>;
            // static constexpr auto this_blas_fill_mode_v = this_blas_fill_mode::value;

            // // Diagonal
            // // * Default value: NONE
            // // * Dummy value: diagonal::non_unit
            // static constexpr bool has_diagonal = has_operator<operator_type::diagonal, description_type>::value;
            // using dummy_default_blas_diagonal  = Diagonal<diagonal::non_unit>;
            // using this_blas_diagonal           = get_or_default_t<operator_type::diagonal, description_type, dummy_default_blas_diagonal>;
            // static constexpr auto this_blas_diagonal_v = this_blas_diagonal::value;

            // // Side
            // // * Default value: NONE
            // // * Dummy value: side::left
            // static constexpr bool has_side         = has_operator<operator_type::side, description_type>::value;
            // using dummy_default_blas_side          = Side<side::left>;
            // using this_blas_side                   = get_or_default_t<operator_type::side, description_type, dummy_default_blas_side>;
            // static constexpr auto this_blas_side_v = this_blas_side::value;

            // // Alignment
            // // * Default value: M, N, K
            // static constexpr bool has_alignment = has_operator<operator_type::alignment, description_type>::value;
            // using default_blas_alignment        = Alignment<this_blas_size_m_v, this_blas_size_n_v, this_blas_size_k_v>;
            // using this_blas_alignment           = get_or_default_t<operator_type::alignment, description_type, default_blas_alignment>;
            // static constexpr auto this_blas_alignment_a = this_blas_alignment::a_alignment;
            // static constexpr auto this_blas_alignment_b = this_blas_alignment::b_alignment;
            // static constexpr auto this_blas_alignment_c = this_blas_alignment::c_alignment;

            // LeadingDimension
            static constexpr bool has_ld = has_operator<operator_type::ld, description_type>::value;
            static constexpr unsigned int default_lda =
                ((!has_transpose_mode || (this_blas_transpose_mode_a == N)) ? this_blas_size_m_v : this_blas_size_k_v);
            static constexpr unsigned int default_ldb =
                ((!has_transpose_mode || (this_blas_transpose_mode_b == N)) ? this_blas_size_k_v : this_blas_size_n_v);
            static constexpr unsigned int default_ldc = this_blas_size_m_v;
#if defined(__NVCC__) && (__CUDACC_VER_MAJOR__ <= 11) && (__CUDACC_VER_MINOR__ <= 5)
            // NVCC 11.4/11.5 workaround for incorrect error:
            // error: ‘constexpr const unsigned int cublasdx::detail::blas_description<...>::default_lda’ is protected within this context
            using dummy_default_blas_ld               = LeadingDimension<1, 1, 1>;
            static constexpr auto this_blas_lda =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::a : default_lda;
            static constexpr auto this_blas_ldb =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::b : default_ldb;
            static constexpr auto this_blas_ldc =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::c : default_ldc;
#elif defined(__NVCC__) && (__CUDACC_VER_MAJOR__ <= 11)
            using default_blas_ld = LeadingDimension<default_lda, default_ldb, default_ldc>;
            using this_blas_ld = get_or_default_t<operator_type::ld, description_type, default_blas_ld>;
            static constexpr auto this_blas_lda = this_blas_ld::a;
            static constexpr auto this_blas_ldb = this_blas_ld::b;
            static constexpr auto this_blas_ldc = this_blas_ld::c;
#else
            // NVCC 12.X.X workaround
            using dummy_default_blas_ld               = LeadingDimension<1, 1, 1>;
            static constexpr auto this_blas_lda =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::a : default_lda;
            static constexpr auto this_blas_ldb =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::b : default_ldb;
            static constexpr auto this_blas_ldc =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::c : default_ldc;
            using this_blas_ld = LeadingDimension<this_blas_lda, this_blas_ldb, this_blas_ldc>;
#endif

            // Number of real elements in each matrix (includes padding)
            static constexpr auto this_blas_a_size = calculate_matrix_size(this_blas_lda,
                                                                           this_blas_size_m_v,
                                                                           this_blas_size_k_v,
                                                                           this_blas_transpose_mode_a);
            static constexpr auto this_blas_b_size = calculate_matrix_size(this_blas_ldb,
                                                                           this_blas_size_k_v,
                                                                           this_blas_size_n_v,
                                                                           this_blas_transpose_mode_b);
            static constexpr auto this_blas_c_size =
                calculate_matrix_size(this_blas_ldc, this_blas_size_m_v, this_blas_size_n_v, N);

            // Logical dimensions
            // rows, cols
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_a_dim {
                default_lda,
                ((this_blas_transpose_mode_a == N) ? this_blas_size_k_v : this_blas_size_m_v)};
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_b_dim {
                default_ldb,
                ((this_blas_transpose_mode_b == N) ? this_blas_size_n_v : this_blas_size_k_v)};
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_c_dim {default_ldc, this_blas_size_n_v};

            // True if description is complete description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            /// ---- Constraints

            // We can only have one of each option

            // Main operators
            static constexpr bool has_one_function       = has_at_most_one_of<operator_type::function, description_type>::value;
            static constexpr bool has_one_precision      = has_at_most_one_of<operator_type::precision, description_type>::value;
            static constexpr bool has_one_size           = has_at_most_one_of<operator_type::size, description_type>::value;
            static constexpr bool has_one_sm             = has_at_most_one_of<operator_type::sm, description_type>::value;
            static constexpr bool has_one_type           = has_at_most_one_of<operator_type::type, description_type>::value;
            static constexpr bool has_one_block_dim      = has_at_most_one_of<operator_type::block_dim, description_type>::value;
            // static constexpr bool has_one_side           = has_at_most_one_of<operator_type::side, description_type>::value;
            // static constexpr bool has_one_diagonal       = has_at_most_one_of<operator_type::diagonal, description_type>::value;
            // static constexpr bool has_one_alignment      = has_at_most_one_of<operator_type::alignment, description_type>::value;
            static constexpr bool has_one_ld             = has_at_most_one_of<operator_type::ld, description_type>::value;
            // static constexpr bool has_one_fill_mode      = has_at_most_one_of<operator_type::fill_mode, description_type>::value;
            static constexpr bool has_one_transpose_mode = has_at_most_one_of<operator_type::transpose_mode, description_type>::value;

            static_assert(has_one_function, "Can't create blas function with two Function<> expressions");
            static_assert(has_one_precision, "Can't create blas function with two Precision<> expressions");
            static_assert(has_one_size, "Can't create blas function with two Size<> expressions");
            static_assert(has_one_sm, "Can't create blas function with two SM<> expressions");
            static_assert(has_one_type, "Can't create blas function with two Type<> expressions");
            static_assert(has_one_block_dim, "Can't create blas function with two BlockDim<> expressions");
            // static_assert(has_one_side, "Can't create blas function with two Side<> expressions");
            // static_assert(has_one_diagonal, "Can't create blas function with two Diagonal<> expressions");
            // static_assert(has_one_alignment, "Can't create blas function with two Alignment<> expressions");
            static_assert(has_one_ld, "Can't create blas function with two LeadingDimension<> expressions");
            // static_assert(has_one_fill_mode, "Can't create blas function with two FillMode<> expressions");
            static_assert(has_one_transpose_mode, "Can't create blas function with two TransposeMode<> expressions");

            // Operators checks

            // // For TRSM FillMode must be upper or lower
            // static constexpr bool valid_trsm_fill_mode =
            //     !has_function ||
            //     !(this_blas_function_v == function::TRSM) ||
            //     !has_fill_mode ||
            //     ((this_blas_fill_mode_v == fill_mode::upper) || (this_blas_fill_mode_v == fill_mode::lower));
            // static_assert(valid_trsm_fill_mode, "Provided fill mode is invalid, for TRSM fill mode must be fill_mode::lower or fill_mode::upper");

            // // For Diagonal and Side can only be defined with TRSM
            // static constexpr bool valid_mm_description_no_trsm_ops =
            //     !has_function ||
            //     !(this_blas_function_v != function::TRSM) ||
            //     !(has_diagonal || has_side);
            // static_assert(valid_mm_description_no_trsm_ops, "For operators Side<> and Diagonal<> can only be used with TRSM function");

            // Leading dimensions check
            // NN --> >=LD(M, K, M)
            // TN --> >=LD(K, K, M)
            // NT --> >=LD(M, N, M)
            // TT --> >=LD(K, N, M)
            static constexpr bool valid_lda =
                !has_ld ||
                !has_size ||
                !has_transpose_mode ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_lda >= default_lda);
            static_assert(valid_lda || (this_blas_transpose_mode_a != N),
                "Incorrect leading dimension for A matrix, LDA must be greater than M");
            static_assert(valid_lda || (this_blas_transpose_mode_a == N),
                "Incorrect leading dimension for A matrix, LDA must be greater than K");
            static constexpr bool valid_ldb =
                !has_ld ||
                !has_size ||
                !has_transpose_mode ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_ldb >= default_ldb);
            static_assert(valid_ldb || (this_blas_transpose_mode_b != N),
                "Incorrect leading dimension for B matrix, LDB must be greater than K");
            static_assert(valid_ldb || (this_blas_transpose_mode_b == N),
                "Incorrect leading dimension for B matrix, LDB must be greater than N");
            static constexpr bool valid_ldc =
                !has_ld ||
                !has_size ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_ldc >= default_ldc);
            static_assert(valid_ldc, "Incorrect leading dimension for C matrix, LDC must be greater than M");

            // Size, precision, type, sm check

            // GEMM
            // Size
            static constexpr bool valid_size_for_block_gemm =
                !has_size ||
                !has_function ||
                !has_sm ||
                !(this_blas_function_v == function::MM) ||
                (is_supported_logical_size<this_blas_precision_t, this_blas_type_v, this_blas_size, this_blas_sm_v>::value);
            static_assert(valid_size_for_block_gemm,
                          "Provided size (M, N, K) for GEMM exceeds maximum supported for selected precision and type. "
                          "Matrices A, B, and C must fit into shared memory.");
            // LD
            static constexpr bool valid_ld_for_block_gemm =
                !has_size ||
                !has_ld ||
                !has_function ||
                !has_sm ||
                !(this_blas_function_v == function::MM) ||
                (is_supported_real_size<this_blas_precision_t, this_blas_type_v, this_blas_a_size, this_blas_b_size, this_blas_c_size, this_blas_sm_v>::value);
            static_assert(valid_ld_for_block_gemm,
                          "Provided leading dimensions for GEMM exceeds maximum supported for selected precision and type. "
                          "Matrices A, B, and C must fit into shared memory.");

            /// ---- End of Constraints
        };

        template<>
        class blas_description<>: public commondx::detail::description_expression {};
    } // namespace detail
} // namespace cublasdx

#undef STRINGIFY
#undef XSTRINGIFY

#endif // CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
