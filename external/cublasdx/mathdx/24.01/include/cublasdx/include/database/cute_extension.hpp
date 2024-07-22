// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_HPP

namespace cute {
    //
    // SM70
    //

    // FP16, 8x8x4, TN
    struct SM70_8x8x4_F16F16F16F16_TN_CUBLASDX: SM70_8x8x4_F16F16F16F16_TN {};

    template<>
    struct MMA_Traits<SM70_8x8x4_F16F16F16F16_TN_CUBLASDX>: MMA_Traits<SM70_8x8x4_F16F16F16F16_TN> {
        using ElementDVal = __half;
        using ElementAVal = __half;
        using ElementBVal = __half;
        using ElementCVal = __half;
    };

    //
    // SM80
    //

    // HMMA, FP16, 16x8x8, TN
    struct SM80_16x8x8_F16F16F16F16_TN_CUBLASDX: SM80_16x8x8_F16F16F16F16_TN {};

    template<>
    struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN_CUBLASDX>: MMA_Traits<SM80_16x8x8_F16F16F16F16_TN> {
        using ElementDVal = __half;
        using ElementAVal = __half;
        using ElementBVal = __half;
        using ElementCVal = __half;
    };

    // HMMA, FP16, 16x8x16, TN
    struct SM80_16x8x16_F16F16F16F16_TN_CUBLASDX: SM80_16x8x16_F16F16F16F16_TN {};

    template<>
    struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN_CUBLASDX>: MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> {
        using ElementDVal = __half;
        using ElementAVal = __half;
        using ElementBVal = __half;
        using ElementCVal = __half;
    };

    // ZMMA, complex fp64, 8x8x4, TN
    struct SM80_8x8x4_C64C64C64C64_TN_CUBLASDX: SM80_8x8x4_C64C64C64C64_TN {};

    template<>
    struct MMA_Traits<SM80_8x8x4_C64C64C64C64_TN_CUBLASDX>: MMA_Traits<SM80_8x8x4_C64C64C64C64_TN> {
        using ElementDVal = cublasdx::complex<double>;
        using ElementAVal = cublasdx::complex<double>;
        using ElementBVal = cublasdx::complex<double>;
        using ElementCVal = cublasdx::complex<double>;
    };


    //
    // SM90
    //

    // ZMMA, complex<fp64>, 16x8x4, TN
    struct SM90_16x8x4_C64C64C64C64_TN_CUBLASDX: SM90_16x8x4_C64C64C64C64_TN {};

    template<>
    struct MMA_Traits<SM90_16x8x4_C64C64C64C64_TN_CUBLASDX>: MMA_Traits<SM90_16x8x4_C64C64C64C64_TN> {
        using ElementDVal = cublasdx::complex<double>;
        using ElementAVal = cublasdx::complex<double>;
        using ElementBVal = cublasdx::complex<double>;
        using ElementCVal = cublasdx::complex<double>;
    };

    // ZMMA, complex<fp64>, 16x8x8, TN
    struct SM90_16x8x8_C64C64C64C64_TN_CUBLASDX: SM90_16x8x8_C64C64C64C64_TN {};

    template<>
    struct MMA_Traits<SM90_16x8x8_C64C64C64C64_TN_CUBLASDX>: MMA_Traits<SM90_16x8x8_C64C64C64C64_TN> {
        using ElementDVal = cublasdx::complex<double>;
        using ElementAVal = cublasdx::complex<double>;
        using ElementBVal = cublasdx::complex<double>;
        using ElementCVal = cublasdx::complex<double>;
    };

    // ZMMA, complex<fp64>, 16x8x16, TN
    struct SM90_16x8x16_C64C64C64C64_TN_CUBLASDX: SM90_16x8x16_C64C64C64C64_TN {};

    template<>
    struct MMA_Traits<SM90_16x8x16_C64C64C64C64_TN_CUBLASDX>: MMA_Traits<SM90_16x8x16_C64C64C64C64_TN> {
        using ElementDVal = cublasdx::complex<double>;
        using ElementAVal = cublasdx::complex<double>;
        using ElementBVal = cublasdx::complex<double>;
        using ElementCVal = cublasdx::complex<double>;
    };

} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_HPP
