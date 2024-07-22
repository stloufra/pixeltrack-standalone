// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_HPP
#define CUBLASDX_DATABASE_CUTE_HPP

#include "cute_tensor.hpp"
#include "cute_db.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

        // CuTe backend details
using cute::Int;
using cute::Layout;
using cute::Shape;
using cute::Tensor;

    // For non-complex types (ie. real type) conjugate does nothing
    template<class T>
    struct conjugate: cute::identity {
    };

    template<class T>
    struct conjugate<cublasdx::complex<T>> {
        __device__ __forceinline__ cublasdx::complex<T> operator()(cublasdx::complex<T> value) const {
            return cublasdx::complex<T> {value.x, -value.y};
        }
    };

    template<>
    struct conjugate<cublasdx::complex<__half>> {
        __device__ __forceinline__ cublasdx::complex<__half> operator()(cublasdx::complex<__half> value) const {
            *reinterpret_cast<unsigned int*>(&value.xy) ^= 0x80000000;
            return value;
        }
    };

template<typename TypeA,
         typename TypeB,
         typename TypeC,
         typename TypeAlpha,
         typename TypeBeta,
         int SizeM,
         int SizeN,
         int SizeK,
         unsigned int LDA,
         unsigned int LDB,
         unsigned int LDC,
         typename TransposeMode,
         typename SM,
         typename BlockSize,
         bool     HasBlockSize,
         bool     Benchmark = false,
         mma_atom MmaAtom   = mma_atom::universal_fma,
         int      TileX     = 0,
         int      TileY     = 0>
struct execution {
    using value_type = TypeC;
    using blas_transpose_mode = TransposeMode;

    static constexpr unsigned int sm = SM::value;

    static constexpr unsigned int m = SizeM;
    static constexpr unsigned int n = SizeN;
    static constexpr unsigned int k = SizeK;

    static constexpr unsigned int lda = LDA;
    static constexpr unsigned int ldb = LDB;
    static constexpr unsigned int ldc = LDC;

    using a_shape_t                 = decltype(make_shape(Int<m> {}, Int<k> {}));
    using non_transposed_a_layout_t = decltype(make_layout(a_shape_t {}, make_stride(Int<1> {}, Int<lda /* m */> {})));
    using transposed_a_layout_t     = decltype(make_layout(a_shape_t {}, make_stride(Int<lda /* k */> {}, Int<1> {})));
    using a_layout_t     = COMMONDX_STL_NAMESPACE::conditional_t<TransposeMode::a_transpose_mode != transpose_mode::non_transposed,
                                          transposed_a_layout_t,
                                          non_transposed_a_layout_t>;
    using a_load_functor = COMMONDX_STL_NAMESPACE::
        conditional_t<TransposeMode::a_transpose_mode == transpose_mode::conj_transposed, conjugate<TypeA>, cute::identity>;

    using b_shape_t                 = decltype(make_shape(Int<n> {}, Int<k> {}));
    using transposed_b_layout_t     = decltype(make_layout(b_shape_t {}, make_stride(Int<1> {}, Int<ldb /* n */> {})));
    using non_transposed_b_layout_t = decltype(make_layout(b_shape_t {}, make_stride(Int<ldb /* k */ > {}, Int<1> {})));
    using b_layout_t     = COMMONDX_STL_NAMESPACE::conditional_t<TransposeMode::b_transpose_mode != transpose_mode::non_transposed,
                                          transposed_b_layout_t,
                                          non_transposed_b_layout_t>;
    using b_load_functor = COMMONDX_STL_NAMESPACE::
        conditional_t<TransposeMode::b_transpose_mode == transpose_mode::conj_transposed, conjugate<TypeB>, cute::identity>;

    using c_shape_t                 = decltype(make_shape(Int<m> {}, Int<n> {}));
    using non_transposed_c_layout_t = decltype(make_layout(c_shape_t {}, make_stride(Int<1> {}, Int<ldc /* m */> {})));
    using transposed_c_layout_t     = decltype(make_layout(c_shape_t {}, make_stride(Int<ldc /* n */> {}, Int<1> {})));
    using c_layout_t                = COMMONDX_STL_NAMESPACE::conditional_t<false, transposed_c_layout_t, non_transposed_c_layout_t>;


    static constexpr unsigned int threads = (BlockSize::value.x * BlockSize::value.y * BlockSize::value.z);
    using mma_t = typename cute_config<TypeA,
                                       TypeC,
                                       m,
                                       n,
                                       k,
                                       TransposeMode,
                                       threads,
                                       HasBlockSize,
                                       SM,
                                       Benchmark,
                                       MmaAtom,
                                       TileX,
                                       TileY>::config;
    static constexpr unsigned int suggested_threads = cute::size(mma_t{});

    __device__ __forceinline__ static unsigned int get_thread_idx() {
        if constexpr (BlockSize::rank == 3) {
            return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        } else if (BlockSize::rank == 2) {
            return threadIdx.x + threadIdx.y * blockDim.x;
        } else {
            return threadIdx.x;
        }
    }

    // STATIC GEMM
    // Strides are assumed to be equal to matrix sizes (depending on transposal)
    // LDA = M / K
    // LDB = K / N
    // LDC = M / N
    __device__ __forceinline__ static void static_gemm(TypeA* A, TypeB* B, TypeC* C, TypeAlpha alpha, TypeBeta beta) {
        const auto thread_idx = get_thread_idx();
        if (thread_idx < cute::size(mma_t {})) {
            Tensor smem_tensor_a = cute::make_tensor(cute::make_smem_ptr(A), a_layout_t {});
            Tensor smem_tensor_b = cute::make_tensor(cute::make_smem_ptr(B), b_layout_t {});
            Tensor smem_tensor_c = cute::make_tensor(cute::make_smem_ptr(C), c_layout_t {});

            mma_t mma;

            auto thr_mma = mma.get_slice(thread_idx);
            cute::gemm(thr_mma, alpha, smem_tensor_a, smem_tensor_b, beta, smem_tensor_c, a_load_functor(), b_load_functor());
        }
    }

    // DYNAMIC GEMM
    // Strides are provided as a runtime argument
    // LDA = lda
    // LDB = ldb
    // LDC = ldc
    __device__ __forceinline__ static void dynamic_gemm(TypeA*       A,
                                                        unsigned int lda,
                                                        TypeB*       B,
                                                        unsigned int ldb,
                                                        TypeC*       C,
                                                        unsigned int ldc,
                                                        TypeAlpha    alpha,
                                                        TypeBeta     beta) {
        const auto thread_idx = get_thread_idx();
        if (thread_idx < cute::size(mma_t {})) {
            constexpr bool is_a_tranposed = (TransposeMode::a_transpose_mode != transpose_mode::non_transposed);
            auto           a_layout_dynamic =
                make_layout(a_shape_t {}, is_a_tranposed ? cute::make_stride(lda, 1u) : cute::make_stride(1u, lda));

            constexpr bool is_b_tranposed = (TransposeMode::b_transpose_mode != transpose_mode::non_transposed);
            auto           b_layout_dynamic =
                make_layout(b_shape_t {}, is_b_tranposed ? cute::make_stride(1u, ldb) : cute::make_stride(ldb, 1u));

            // C is assumed to always be non-transposed
            auto c_layout_dynamic = make_layout(c_shape_t {}, cute::make_stride(1, ldc));

            Tensor smem_tensor_a = cute::make_tensor(cute::make_smem_ptr(A), a_layout_dynamic);
            Tensor smem_tensor_b = cute::make_tensor(cute::make_smem_ptr(B), b_layout_dynamic);
            Tensor smem_tensor_c = cute::make_tensor(cute::make_smem_ptr(C), c_layout_dynamic);

            mma_t mma;

            auto thr_mma = mma.get_slice(thread_idx);
            cute::gemm(thr_mma, alpha, smem_tensor_a, smem_tensor_b, beta, smem_tensor_c, a_load_functor(), b_load_functor());
        }
    }
};

template<typename TypeA,
         typename TypeB,
         typename TypeC,
         typename TypeAlpha,
         typename TypeBeta,
         int SizeM,
         int SizeN,
         int SizeK,
         unsigned int LDA,
         unsigned int LDB,
         unsigned int LDC,
         typename TransposeMode,
         typename SM>
struct execution_suggestions {
private:
    using execution_type = cute_backend::execution<TypeA,
                                                   TypeB,
                                                   TypeC,
                                                   TypeAlpha,
                                                   TypeBeta,
                                                   SizeM,
                                                   SizeN,
                                                   SizeK,
                                                   LDA,
                                                   LDB,
                                                   LDC,
                                                   TransposeMode,
                                                   SM,
                                                   BlockDim<256> /* dummy */,
                                                   false>;

public:
    using block_dim = BlockDim<execution_type::suggested_threads, 1, 1>;
};

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_HPP
