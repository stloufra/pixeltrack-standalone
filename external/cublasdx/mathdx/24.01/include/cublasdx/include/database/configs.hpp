// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CONFIGS_HPP
#define CUBLASDX_DATABASE_CONFIGS_HPP

#include "commondx/detail/stl/tuple.hpp"

#include "cute_tensor_configs.hpp"
#include "commondx/type_list.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {
            namespace database {

enum transposition_abbreviation
{
    NN,
    NT,
    TN,
    TT
};

template<int BlockDim, mma_atom MMAAtom, int TileX, int TileY>
struct generated_config_impl {
    static constexpr mma_atom mma      = MMAAtom;
    static constexpr auto     tiles    = COMMONDX_STL_NAMESPACE::make_tuple(TileX, TileY);
    static constexpr int      blockdim = BlockDim;
};

template<typename... ListElements>
struct generated_config_list {
    static constexpr bool defined = true;
    using list                    = commondx::type_list<ListElements...>;
};

template<typename InputType,
         typename OutputType,
         int                        M,
         int                        N,
         int                        K,
         transposition_abbreviation Transposition,
         typename SM>
struct generated_config {
    static constexpr bool defined = false;
};

#define CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1(SMValue, Transposition)                       \
    template<typename InputType, typename OutputType>                                         \
    struct generated_config<InputType, OutputType, 4, 4, 4, Transposition, SM<SMValue>>:      \
        generated_config_list<generated_config_impl<64, mma_atom::universal_fma, 4, 16>,      \
                              generated_config_impl<128, mma_atom::universal_fma, 4, 32>,     \
                              generated_config_impl<256, mma_atom::universal_fma, 4, 64>,     \
                              generated_config_impl<512, mma_atom::universal_fma, 4, 128>,    \
                              generated_config_impl<1024, mma_atom::universal_fma, 4, 256>> { \
    };                                                                                        \
    template<typename InputType, typename OutputType>                                         \
    struct generated_config<InputType, OutputType, 8, 8, 8, Transposition, SM<SMValue>>:      \
        generated_config_list<generated_config_impl<64, mma_atom::universal_fma, 8, 8>,       \
                              generated_config_impl<128, mma_atom::universal_fma, 8, 16>,     \
                              generated_config_impl<256, mma_atom::universal_fma, 8, 32>,     \
                              generated_config_impl<512, mma_atom::universal_fma, 8, 64>,     \
                              generated_config_impl<1024, mma_atom::universal_fma, 8, 128>> { \
    };

#define CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(SMValue)   \
    CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1(SMValue, NN) \
    CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1(SMValue, NT) \
    CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1(SMValue, TN) \
    CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1(SMValue, TT)

CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(700)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(720)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(750)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(800)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(860)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(870)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(890)
CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS(900)

#undef CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS
#undef CUBLASDX_DETAIL_CONFIG_MANUAL_CONFIGS_1

// SM70
#include "sm70/fp16_nn.hpp.inc"
#include "sm70/fp32_nn.hpp.inc"
#include "sm70/fp64_nn.hpp.inc"
#include "sm70/cfp16_nn.hpp.inc"
#include "sm70/cfp32_nn.hpp.inc"
#include "sm70/cfp64_nn.hpp.inc"

#include "sm70/fp16_nt.hpp.inc"
#include "sm70/fp32_nt.hpp.inc"
#include "sm70/fp64_nt.hpp.inc"
#include "sm70/cfp16_nt.hpp.inc"
#include "sm70/cfp32_nt.hpp.inc"
#include "sm70/cfp64_nt.hpp.inc"

#include "sm70/fp16_tn.hpp.inc"
#include "sm70/fp32_tn.hpp.inc"
#include "sm70/fp64_tn.hpp.inc"
#include "sm70/cfp16_tn.hpp.inc"
#include "sm70/cfp32_tn.hpp.inc"
#include "sm70/cfp64_tn.hpp.inc"

#include "sm70/fp16_tt.hpp.inc"
#include "sm70/fp32_tt.hpp.inc"
#include "sm70/fp64_tt.hpp.inc"
#include "sm70/cfp16_tt.hpp.inc"
#include "sm70/cfp32_tt.hpp.inc"
#include "sm70/cfp64_tt.hpp.inc"

// SM75
// No database for FP64 due to low performance
#include "sm75/fp16_nn.hpp.inc"
#include "sm75/fp32_nn.hpp.inc"
#include "sm75/cfp16_nn.hpp.inc"
#include "sm75/cfp32_nn.hpp.inc"

#include "sm75/fp16_nt.hpp.inc"
#include "sm75/fp32_nt.hpp.inc"
#include "sm75/cfp16_nt.hpp.inc"
#include "sm75/cfp32_nt.hpp.inc"

#include "sm75/fp16_tn.hpp.inc"
#include "sm75/fp32_tn.hpp.inc"
#include "sm75/cfp16_tn.hpp.inc"
#include "sm75/cfp32_tn.hpp.inc"

#include "sm75/fp16_tt.hpp.inc"
#include "sm75/fp32_tt.hpp.inc"
#include "sm75/cfp16_tt.hpp.inc"
#include "sm75/cfp32_tt.hpp.inc"

// SM80
#include "sm80/fp16_nn.hpp.inc"
#include "sm80/fp32_nn.hpp.inc"
#include "sm80/fp64_nn.hpp.inc"
#include "sm80/cfp16_nn.hpp.inc"
#include "sm80/cfp32_nn.hpp.inc"
#include "sm80/cfp64_nn.hpp.inc"

#include "sm80/fp16_nt.hpp.inc"
#include "sm80/fp32_nt.hpp.inc"
#include "sm80/fp64_nt.hpp.inc"
#include "sm80/cfp16_nt.hpp.inc"
#include "sm80/cfp32_nt.hpp.inc"
#include "sm80/cfp64_nt.hpp.inc"

#include "sm80/fp16_tn.hpp.inc"
#include "sm80/fp32_tn.hpp.inc"
#include "sm80/fp64_tn.hpp.inc"
#include "sm80/cfp16_tn.hpp.inc"
#include "sm80/cfp32_tn.hpp.inc"
#include "sm80/cfp64_tn.hpp.inc"

#include "sm80/fp16_tt.hpp.inc"
#include "sm80/fp32_tt.hpp.inc"
#include "sm80/fp64_tt.hpp.inc"
#include "sm80/cfp16_tt.hpp.inc"
#include "sm80/cfp32_tt.hpp.inc"
#include "sm80/cfp64_tt.hpp.inc"

// SM86
// No database for FP64 due to low performance
#include "sm86/fp16_nn.hpp.inc"
#include "sm86/fp32_nn.hpp.inc"
#include "sm86/cfp16_nn.hpp.inc"
#include "sm86/cfp32_nn.hpp.inc"

#include "sm86/fp16_nt.hpp.inc"
#include "sm86/fp32_nt.hpp.inc"
#include "sm86/cfp16_nt.hpp.inc"
#include "sm86/cfp32_nt.hpp.inc"

#include "sm86/fp16_tn.hpp.inc"
#include "sm86/fp32_tn.hpp.inc"
#include "sm86/cfp16_tn.hpp.inc"
#include "sm86/cfp32_tn.hpp.inc"

#include "sm86/fp16_tt.hpp.inc"
#include "sm86/fp32_tt.hpp.inc"
#include "sm86/cfp16_tt.hpp.inc"
#include "sm86/cfp32_tt.hpp.inc"

// SM89
// No database for FP64 due to low performance
#include "sm89/fp16_nn.hpp.inc"
#include "sm89/fp32_nn.hpp.inc"
#include "sm89/cfp16_nn.hpp.inc"
#include "sm89/cfp32_nn.hpp.inc"

#include "sm89/fp16_nt.hpp.inc"
#include "sm89/fp32_nt.hpp.inc"
#include "sm89/cfp16_nt.hpp.inc"
#include "sm89/cfp32_nt.hpp.inc"

#include "sm89/fp16_tn.hpp.inc"
#include "sm89/fp32_tn.hpp.inc"
#include "sm89/cfp16_tn.hpp.inc"
#include "sm89/cfp32_tn.hpp.inc"

#include "sm89/fp16_tt.hpp.inc"
#include "sm89/fp32_tt.hpp.inc"
#include "sm89/cfp16_tt.hpp.inc"
#include "sm89/cfp32_tt.hpp.inc"

// SM90
#include "sm90/fp16_nn.hpp.inc"
#include "sm90/fp32_nn.hpp.inc"
#include "sm90/fp64_nn.hpp.inc"
#include "sm90/cfp16_nn.hpp.inc"
#include "sm90/cfp32_nn.hpp.inc"
#include "sm90/cfp64_nn.hpp.inc"

#include "sm90/fp16_nt.hpp.inc"
#include "sm90/fp32_nt.hpp.inc"
#include "sm90/fp64_nt.hpp.inc"
#include "sm90/cfp16_nt.hpp.inc"
#include "sm90/cfp32_nt.hpp.inc"
#include "sm90/cfp64_nt.hpp.inc"

#include "sm90/fp16_tn.hpp.inc"
#include "sm90/fp32_tn.hpp.inc"
#include "sm90/fp64_tn.hpp.inc"
#include "sm90/cfp16_tn.hpp.inc"
#include "sm90/cfp32_tn.hpp.inc"
#include "sm90/cfp64_tn.hpp.inc"

#include "sm90/fp16_tt.hpp.inc"
#include "sm90/fp32_tt.hpp.inc"
#include "sm90/fp64_tt.hpp.inc"
#include "sm90/cfp16_tt.hpp.inc"
#include "sm90/cfp32_tt.hpp.inc"
#include "sm90/cfp64_tt.hpp.inc"

// Forward SM72 to SM70
template<typename InputType, typename OutputType, int M, int N, int K, transposition_abbreviation Transposition>
struct generated_config<InputType, OutputType, M, N, K, Transposition, SM<720>>:
    public generated_config<InputType, OutputType, M, N, K, Transposition, SM<700>> {
};

// Forward SM87 to SM80
template<typename InputType, typename OutputType, int M, int N, int K, transposition_abbreviation Transposition>
struct generated_config<InputType, OutputType, M, N, K, Transposition, SM<870>>:
    public generated_config<InputType, OutputType, M, N, K, Transposition, SM<800>> {
};

            } // namespace database
        }     // namespace cute_backend
    }         // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CONFIGS_HPP
