// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "commondx/traits/detail/make_cudavalue_type.hpp"

#include "blas_description.hpp"
#include "database/cute.hpp"

#include "../traits.hpp"

namespace cublasdx {
    namespace detail {
        using commondx::detail::make_cudavalue_type_t;

        template<class... Operators>
        class blas_execution: public blas_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type = blas_description<Operators...>;
            using this_type = blas_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_blas_precision_t;

            /// ---- Constraints

            // We need Block operator to be specified exactly once
            static constexpr bool has_one_block = has_at_most_one_of<operator_type::block, this_type>::value;
            static_assert(has_one_block, "Can't create blas function with two execution operators");
        };

#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
        template<class... Operators>
        class blas_block_execution_partial: public blas_execution<Operators...>
        {
            using base_type = blas_execution<Operators...>;
            using typename base_type::this_blas_precision_t;
            using this_blas_value_type =
                COMMONDX_STL_NAMESPACE::conditional_t<(base_type::this_blas_type_v == type::complex),
                                                      complex<this_blas_precision_t>,
                                                      this_blas_precision_t>;

        public:
            using value_type  = this_blas_value_type;
            using input_type  = value_type;
            using output_type = value_type;
        };
#endif

        template<class... Operators>
        class blas_block_execution: public blas_execution<Operators...>
        {
            using this_type = blas_block_execution<Operators...>;
            using base_type = blas_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_blas_precision_t;

            // Value type
            using this_blas_value_type =
                COMMONDX_STL_NAMESPACE::conditional_t<(base_type::this_blas_type_v == type::complex),
                                                      complex<this_blas_precision_t>,
                                                      this_blas_precision_t>;

            /// ---- Suggestions
            using execution_suggestions =
                cute_backend::execution_suggestions<this_blas_value_type,
                                                    this_blas_value_type,
                                                    this_blas_value_type,
                                                    this_blas_value_type,
                                                    this_blas_value_type,
                                                    base_type::this_blas_size_m_v,
                                                    base_type::this_blas_size_n_v,
                                                    base_type::this_blas_size_k_v,
                                                    base_type::this_blas_lda,
                                                    base_type::this_blas_ldb,
                                                    base_type::this_blas_ldc,
                                                    typename base_type::this_blas_transpose_mode,
                                                    typename base_type::this_blas_sm>;

            /// ---- Traits

            // Block Dimension
            // * Default value: selected by implementation
            static constexpr bool has_block_dim = has_operator<operator_type::block_dim, base_type>::value;
            using default_blas_block_dim        = typename execution_suggestions::block_dim;
            using this_blas_block_dim           = get_or_default_t<operator_type::block_dim, base_type, default_blas_block_dim>;
            static constexpr auto this_blas_block_dim_v = this_blas_block_dim::value;

            /// ---- Checks

            static constexpr bool valid_block_dim = this_blas_block_dim::flat_size >= 32 && this_blas_block_dim::flat_size <= 1024;
            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't "
                          "have more than 1024 threads.");

            /// ---- Backend

            // CuTe backend implementation
            using execution_type = cute_backend::execution<this_blas_value_type,
                                                           this_blas_value_type,
                                                           this_blas_value_type,
                                                           this_blas_value_type,
                                                           this_blas_value_type,
                                                           base_type::this_blas_size_m_v,
                                                           base_type::this_blas_size_n_v,
                                                           base_type::this_blas_size_k_v,
                                                           base_type::this_blas_lda,
                                                           base_type::this_blas_ldb,
                                                           base_type::this_blas_ldc,
                                                           typename base_type::this_blas_transpose_mode,
                                                           typename base_type::this_blas_sm,
                                                           this_blas_block_dim,
                                                           this_type::has_block_dim>;


        public:
            using value_type  = this_blas_value_type;
            using input_type  = value_type;
            using output_type = value_type;

            inline __device__ void execute(const value_type   alpha,
                                           value_type*        matrix_a,
                                           const unsigned int lda,
                                           value_type*        matrix_b,
                                           const unsigned int ldb,
                                           const value_type   beta,
                                           value_type*        matrix_c,
                                           const unsigned int ldc) {
                execution_type::dynamic_gemm(matrix_a, lda, matrix_b, ldb, matrix_c, ldc, alpha, beta);
            }

            inline __device__ void execute(const value_type   alpha,
                                value_type*        matrix_a,
                                value_type*        matrix_b,
                                const value_type   beta,
                                value_type*        matrix_c) {
                execution_type::static_gemm(matrix_a, matrix_b, matrix_c, alpha, beta);
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T>
            inline __device__ auto execute(const T            alpha,
                                           T*                 matrix_a,
                                           const unsigned int lda,
                                           T*                 matrix_b,
                                           const unsigned int ldb,
                                           const T            beta,
                                           T*                 matrix_c,
                                           const unsigned int ldc) //
                -> typename COMMONDX_STL_NAMESPACE::enable_if<!COMMONDX_STL_NAMESPACE::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                execution_type::dynamic_gemm(reinterpret_cast<value_type*>(matrix_a),
                                             lda,
                                             reinterpret_cast<value_type*>(matrix_b),
                                             ldb,
                                             reinterpret_cast<value_type*>(matrix_c),
                                             ldc,
                                             *reinterpret_cast<const value_type*>(&alpha),
                                             *reinterpret_cast<const value_type*>(&beta));
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T>
            inline __device__ auto execute(const T            alpha,
                                           T*                 matrix_a,
                                           T*                 matrix_b,
                                           const T            beta,
                                           T*                 matrix_c) //
                -> typename COMMONDX_STL_NAMESPACE::enable_if<!COMMONDX_STL_NAMESPACE::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                execution_type::static_gemm(reinterpret_cast<value_type*>(matrix_a),
                                            reinterpret_cast<value_type*>(matrix_b),
                                            reinterpret_cast<value_type*>(matrix_c),
                                            *reinterpret_cast<const value_type*>(&alpha),
                                            *reinterpret_cast<const value_type*>(&beta));
            }

            template<class T>
            inline __device__ auto execute(const T /* alpha */,
                                           T* /* matrix_a */,
                                           const unsigned int /* lda */,
                                           T* /* matrix_b */,
                                           const unsigned int /* ldb */,
                                           const T /* beta */,
                                           T* /* matrix_c */,
                                           const unsigned int /* ldc */) //
                -> typename COMMONDX_STL_NAMESPACE::enable_if<COMMONDX_STL_NAMESPACE::is_void<T>::value || (sizeof(T) != sizeof(value_type))
                    || (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    COMMONDX_STL_NAMESPACE::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

            template<class T>
            inline __device__ auto execute(const T /* alpha */,
                                           T* /* matrix_a */,
                                           T* /* matrix_b */,
                                           const T /* beta */,
                                           T* /* matrix_c */) //
                -> typename COMMONDX_STL_NAMESPACE::enable_if<COMMONDX_STL_NAMESPACE::is_void<T>::value || (sizeof(T) != sizeof(value_type))
                    || (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    COMMONDX_STL_NAMESPACE::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

        private:
            inline static constexpr dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete, "Can't provide suggested block dimensions, description is not complete");
                return execution_suggestions::block_dim::value;
            }

            inline static constexpr dim3 get_block_dim() {
                static_assert(base_type::is_complete, "Can't provide block dimensions, description is not complete");
                if constexpr(has_block_dim) {
                    return this_blas_block_dim_v;
                }
                return get_suggested_block_dim();
            }

        public:
            inline static constexpr unsigned int get_shared_memory_size() {
                static_assert(base_type::is_complete, "Can't calculate shared memory, description is not complete");
                return sizeof(value_type) * (base_type::this_blas_a_size + base_type::this_blas_b_size + base_type::this_blas_c_size);
            }

            inline static constexpr unsigned int get_shared_memory_size(unsigned int lda, unsigned int ldb, unsigned int ldc) {
                static_assert(base_type::is_complete, "Can't calculate shared memory, description is not complete");
                auto a = calculate_matrix_size(lda, base_type::this_blas_size_m_v, base_type::this_blas_size_k_v, base_type::this_blas_transpose_mode_a);
                auto b = calculate_matrix_size(ldb, base_type::this_blas_size_k_v, base_type::this_blas_size_n_v, base_type::this_blas_transpose_mode_b);
                auto c = calculate_matrix_size(ldc, base_type::this_blas_size_m_v, base_type::this_blas_size_n_v, N);
                return sizeof(value_type) * (a + b + c);
            }

            // Number of elements in A, B, C matrices (includes padding / leading dimensions)
            // (ld * cols)
            static constexpr unsigned int a_size = base_type::this_blas_a_size;
            static constexpr unsigned int b_size = base_type::this_blas_b_size;
            static constexpr unsigned int c_size = base_type::this_blas_c_size;

            // Leading dimensions of A, B, C matrices
            static constexpr unsigned int lda = base_type::this_blas_lda;
            static constexpr unsigned int ldb = base_type::this_blas_ldb;
            static constexpr unsigned int ldc = base_type::this_blas_ldc;

            // Logical dimensions of A, B, C matrices
            // (row; cols)
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> a_dim = base_type::this_blas_a_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> b_dim = base_type::this_blas_b_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> c_dim = base_type::this_blas_c_dim;

            static constexpr dim3         suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3         block_dim           = get_block_dim();
            static constexpr unsigned int shared_memory_size  = get_shared_memory_size();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
            static constexpr unsigned int min_blocks_per_multiprocessor = 1;
        };


        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::a_size;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::b_size;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::c_size;

        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::lda;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::ldb;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::ldc;

        template<class... Operators>
        constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> blas_block_execution<Operators...>::a_dim;
        template<class... Operators>
        constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> blas_block_execution<Operators...>::b_dim;
        template<class... Operators>
        constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> blas_block_execution<Operators...>::c_dim;

        template<class... Operators>
        constexpr dim3 blas_block_execution<Operators...>::suggested_block_dim;
        template<class... Operators>
        constexpr dim3 blas_block_execution<Operators...>::block_dim;

        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::shared_memory_size;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::max_threads_per_block;
        template<class... Operators>
        constexpr unsigned int blas_block_execution<Operators...>::min_blocks_per_multiprocessor;

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator     = has_operator<operator_type::block, blas_operator_wrapper<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called fft_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before FFT description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using operator_wrapper_type = blas_operator_wrapper<Operators...>;
            using execution_type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_complete_blas<operator_wrapper_type>::value,
                                                             blas_block_execution<Operators...>,
                                                             blas_block_execution_partial<Operators...>>::type;
#else
            using execution_type = blas_block_execution<Operators...>;
#endif
            using description_type = blas_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::blas_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::blas_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::blas_description<Operators1...>&,
                                                       const detail::blas_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
