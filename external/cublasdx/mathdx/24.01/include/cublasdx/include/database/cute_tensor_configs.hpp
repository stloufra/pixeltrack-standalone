// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
#define CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP

#include "cute_tensor.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

            // This list is currently incomplete,for extensive testing add all cute::MMA types here
            enum class mma_atom
            {
                universal_fma,
                SM70_8x8x4_F16F16F16F16_TN_CUBLASDX,
                SM80_16x8x8_F16F16F16F16_TN_CUBLASDX,
                SM80_16x8x16_F16F16F16F16_TN_CUBLASDX,
                SM80_8x8x4_F64F64F64F64_TN,
                SM80_8x8x4_C64C64C64C64_TN_CUBLASDX,
                SM90_16x8x4_F64F64F64F64_TN,
                SM90_16x8x8_F64F64F64F64_TN,
                SM90_16x8x16_F64F64F64F64_TN,
                SM90_16x8x4_C64C64C64C64_TN_CUBLASDX,
                SM90_16x8x8_C64C64C64C64_TN_CUBLASDX,
                SM90_16x8x16_C64C64C64C64_TN_CUBLASDX
            };

            template<typename DataType, unsigned int SM>
            constexpr mma_atom get_default_mma() {
                mma_atom ret = mma_atom::universal_fma;
                // 70, 72, 75
                if constexpr (SM < 800) {
                    if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half>) {
                        ret = mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX;
                    }
                // 80, 86, 87, 89
                } else if constexpr (SM < 900) {
                    if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half>) {
                        ret = mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX;
                    } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, double>) {
                        ret = mma_atom::SM80_8x8x4_F64F64F64F64_TN;
                    } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>) {
                        ret = mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX;
                    }
                }
                // 90
                else if constexpr (SM == 900) {
                    if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half>) {
                        ret = mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX;
                    } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, double>) {
                        ret = mma_atom::SM90_16x8x4_F64F64F64F64_TN;
                    } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>) {
                        ret = mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX;
                    }
                }
                return ret;
            }

            // This function returns instance of CuTe type
            // can be used with decltype to get just the type
            template<typename DataType, mma_atom MmaAtom>
            constexpr auto convert_mma_atom_to_cute() {
                if constexpr (MmaAtom == mma_atom::universal_fma) {
                    return cute::UniversalFMA<DataType, DataType, DataType> {};
                }
                if constexpr (MmaAtom == mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX) {
                    return cute::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x8_F16F16F16F16_TN_CUBLASDX) {
                    return cute::SM80_16x8x8_F16F16F16F16_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX) {
                    return cute::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_8x8x4_F64F64F64F64_TN) {
                    return cute::SM80_8x8x4_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX) {
                    return cute::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x4_F64F64F64F64_TN) {
                    return cute::SM90_16x8x4_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x8_F64F64F64F64_TN) {
                    return cute::SM90_16x8x8_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x16_F64F64F64F64_TN) {
                    return cute::SM90_16x8x16_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX) {
                    return cute::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x8_C64C64C64C64_TN_CUBLASDX) {
                    return cute::SM90_16x8x8_C64C64C64C64_TN_CUBLASDX {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x16_C64C64C64C64_TN_CUBLASDX) {
                    return cute::SM90_16x8x16_C64C64C64C64_TN_CUBLASDX {};
                }
            }
        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
