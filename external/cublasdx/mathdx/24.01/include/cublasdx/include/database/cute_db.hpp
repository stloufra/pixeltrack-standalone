// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_DB_HPP
#define CUBLASDX_DATABASE_CUTE_DB_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "commondx/device_info.hpp"
#include "commondx/type_list.hpp"

#include "cute_tensor.hpp"
#include "cute_tensor_configs.hpp"
#include "configs.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

// Selects generated_config from commondx::type_list based on blockdim,
// if there is no such implementation in list search_by_blockdim::type is set to void.
template<int ThreadsAvailable, typename ImplementationList>
struct search_by_blockdim;

template<int ThreadsAvailable, typename GeneratedConfig>
struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig>> {
    using type = COMMONDX_STL_NAMESPACE::conditional_t<GeneratedConfig::blockdim == ThreadsAvailable, GeneratedConfig, void>;
};

template<int ThreadsAvailable, typename GeneratedConfig, typename... RemainingConfigs>
struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig, RemainingConfigs...>> {
    using type = COMMONDX_STL_NAMESPACE::conditional_t<
        GeneratedConfig::blockdim == ThreadsAvailable,
        GeneratedConfig,
        typename search_by_blockdim<ThreadsAvailable, commondx::type_list<RemainingConfigs...>>::type>;
};

// Source: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
constexpr int closest_power_of_2(int value) {
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return ++value;
}

constexpr int closest_multiple_of_16(int value) {
    int remainder = value % 16;
    if (remainder == 0)
        return value;
    return value + 16 - remainder;
}

template<bool HasBlockDim, int ThreadsAvailable, typename ConfigList>
using search_struct = COMMONDX_STL_NAMESPACE::conditional_t<HasBlockDim,
                                         search_by_blockdim<ThreadsAvailable, ConfigList>,
                                         commondx::type_list_element<0, ConfigList>>;

// RETURN TYE STD::TUPLE CONTAINS 3 ELEMENTS:
// 1. BOOL, Whether this is a mma config using tensor cores
// 2. INT, TileX size
// 3. INT, TileY size
template<typename InputType,
         typename OutputType,
         int                                  M,
         int                                  N,
         int                                  K,
         database::transposition_abbreviation Transposition,
         int                                  ThreadsAvailable,
         bool                                 HasBlockDim,
         typename SM>
constexpr auto get_tile_config() {
    #ifdef __NVCOMPILER
    #pragma diag_suppress 185
    #pragma diag_suppress 111
    #endif
    using direct_config_list = database::generated_config<InputType, OutputType, M, N, K, Transposition, SM>;
    if constexpr (direct_config_list::defined) {
        using direct_config =
            typename search_struct<HasBlockDim, ThreadsAvailable, typename direct_config_list::list>::type;
        if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<direct_config, void>) {
            constexpr auto inner_tiles = direct_config::tiles;
            return COMMONDX_STL_NAMESPACE::make_tuple(direct_config::mma, COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles));
        }
    }

    using mnn_config_list = database::generated_config<InputType, OutputType, M, N, N, Transposition, SM>;
    if constexpr (mnn_config_list::defined) {
        using mnn_config = typename search_struct<HasBlockDim, ThreadsAvailable, typename mnn_config_list::list>::type;
        if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<mnn_config, void>) {
            constexpr auto inner_tiles = mnn_config::tiles;
            return COMMONDX_STL_NAMESPACE::make_tuple(mnn_config::mma, COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles));
        }
    }

    using mmm_config_list = database::generated_config<InputType, OutputType, M, M, M, Transposition, SM>;
    if constexpr (mmm_config_list::defined) {
        using mmm_config = typename search_struct<HasBlockDim, ThreadsAvailable, typename mmm_config_list::list>::type;
        if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<mmm_config, void>) {
            constexpr auto inner_tiles = mmm_config::tiles;
            return COMMONDX_STL_NAMESPACE::make_tuple(mmm_config::mma, COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles));
        }
    }

    constexpr int m_rounded = (M < 4) ? 4 : (M < 8) ? 8 : closest_multiple_of_16(M);
    constexpr int n_rounded = (N < 4) ? 4 : (N < 8) ? 8 : closest_multiple_of_16(N);
    constexpr int k_rounded = (K < 4) ? 4 : (K < 8) ? 8 : closest_multiple_of_16(K);

    using rounded_config_list =
        database::generated_config<InputType, OutputType, m_rounded, n_rounded, k_rounded, Transposition, SM>;
    if constexpr (rounded_config_list::defined) {
        using rounded_config =
            typename search_struct<HasBlockDim, ThreadsAvailable, typename rounded_config_list::list>::type;
        if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<rounded_config, void>) {
            constexpr auto inner_tiles = rounded_config::tiles;
            return COMMONDX_STL_NAMESPACE::make_tuple(rounded_config::mma, COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles));
        }
    }

    using rounded_all_m_config_list =
        database::generated_config<InputType, OutputType, m_rounded, m_rounded, m_rounded, Transposition, SM>;
    if constexpr (rounded_all_m_config_list::defined) {
        using rounded_all_m_config =
            typename search_struct<HasBlockDim, ThreadsAvailable, typename rounded_all_m_config_list::list>::type;
        if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<rounded_all_m_config, void>) {
            constexpr auto inner_tiles = rounded_all_m_config::tiles;
            return COMMONDX_STL_NAMESPACE::make_tuple(rounded_all_m_config::mma, COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles));
        }
    }


    // TODO: Remove magic number, remove default BlockDim from blas_description etc.
    constexpr int  heuristic_threads  = HasBlockDim ? ThreadsAvailable : 256;
    constexpr auto default_mma        = get_default_mma<OutputType, SM::value>();
    using default_mma_t               = decltype(convert_mma_atom_to_cute<OutputType, default_mma>());
    constexpr int default_mma_divisor = decltype(cute::size(typename cute::MMA_Traits<default_mma_t>::ThrID {}))::value;
    // If there's no chance matching to default_mma fallback to fma
    constexpr auto selected_mma = ((heuristic_threads % default_mma_divisor) == 0) ? default_mma : mma_atom::universal_fma;
    using selected_mma_t  = decltype(convert_mma_atom_to_cute<OutputType, selected_mma>());
    constexpr int divisor = decltype(cute::size(typename cute::MMA_Traits<selected_mma_t>::ThrID {}))::value;

    int tmp_x = 1;
    for (int iter = 1; iter <= heuristic_threads / divisor; ++iter) {
        tmp_x += 1;
        int tmp_y = (heuristic_threads / divisor) / tmp_x;
        // This loop looks for such a division of ThreadsAvailable into TileX
        // and TileY that ThreadsAvailable == TileX * TileY and TileX > TileY
        // keeping TileX as small as possible. Ideally it will find square root
        // of ThreadsAvailable, but otherwise the most square combination
        // possible
        if (tmp_x >= tmp_y && tmp_x * tmp_y == (heuristic_threads / divisor)) {
            // This makes sure ThrID shape is divisible by the tile (tmp_x, tmp_y) we return
            if constexpr (cute::MMA_Traits<selected_mma_t>::ThrID::rank == 2) {
                constexpr auto tile_x_divisor = cute::size<0>(typename cute::MMA_Traits<selected_mma_t>::ThrID {});
                constexpr auto tile_y_divisor = cute::size<1>(typename cute::MMA_Traits<selected_mma_t>::ThrID {});
                bool           tile_x_correct = ((tmp_x % tile_x_divisor) == 0);
                bool           tile_y_correct = ((tmp_y % tile_y_divisor) == 0);
                if (tile_x_correct && tile_y_correct) {
                    return COMMONDX_STL_NAMESPACE::make_tuple(selected_mma, tmp_x, tmp_y);
                }
                // Maybe if we swap tmp_x <-> tmp_y
                bool swapped_tile_x_correct = ((tmp_y % tile_x_divisor) == 0);
                bool swapped_tile_y_correct = ((tmp_x % tile_y_divisor) == 0);
                if (swapped_tile_x_correct && swapped_tile_y_correct) {
                    return COMMONDX_STL_NAMESPACE::make_tuple(selected_mma, tmp_y, tmp_x);
                }
            } else {
                return COMMONDX_STL_NAMESPACE::make_tuple(selected_mma, tmp_x, tmp_y);
            }
        }
    }
    #ifdef __NVCOMPILER
    #pragma diag_warning 185
    #pragma diag_warning 111
    #endif

    // Final fallback
    return COMMONDX_STL_NAMESPACE::make_tuple(mma_atom::universal_fma, 1, ThreadsAvailable);
}

template<typename TransposeMode>
constexpr database::transposition_abbreviation get_transposition_abbreviation() {
    constexpr bool a_trans = (TransposeMode::a_transpose_mode != transpose_mode::non_transposed);
    constexpr bool b_trans = (TransposeMode::b_transpose_mode != transpose_mode::non_transposed);

    if constexpr (a_trans && b_trans) {
        // Also covers CC
        return database::transposition_abbreviation::TT;
    } else if constexpr (!a_trans && b_trans) {
        // Also covers NC
        return database::transposition_abbreviation::NT;
    } else if constexpr (a_trans && !b_trans) {
        // Also covers CN
        return database::transposition_abbreviation::TN;
    } else if constexpr (!a_trans && !b_trans) {
        return database::transposition_abbreviation::NN;
    }
    return {};
}

template<typename InputType,
         typename OutputType,
         int M,
         int N,
         int K,
         typename TransposeMode,
         int  ThreadsAvailable,
         bool HasBlockSize,
         typename SM>
struct generated_config_getter {
    // if defined then use the defined one
    static constexpr database::transposition_abbreviation transposition =
        get_transposition_abbreviation<TransposeMode>();
    static constexpr auto tile_config =
        get_tile_config<InputType, OutputType, M, N, K, transposition, ThreadsAvailable, HasBlockSize, SM>();
    using config = decltype(cute::make_tiled_mma(
        convert_mma_atom_to_cute<OutputType, COMMONDX_STL_NAMESPACE::get<0>(tile_config)>(),
        cute::Layout<cute::Shape<cute::Int<COMMONDX_STL_NAMESPACE::get<1>(tile_config)>, cute::Int<COMMONDX_STL_NAMESPACE::get<2>(tile_config)>>> {}));
};

template<typename InputType,
         typename OutputType,
         int M,
         int N,
         int K,
         typename TransposeMode,
         int  ThreadsAvailable,
         bool HasBlockSize,
         typename SM,
         bool     Benchmark = false, // Overrides database/heuristic parameters with (MmaAtom, TileX, TileY)
         mma_atom MmaAtom   = mma_atom::universal_fma, // Override
         int      TileX     = 0, // Override
         int      TileY     = 0> // Override
struct cute_config {
    using manual    = decltype(cute::make_tiled_mma(convert_mma_atom_to_cute<OutputType, MmaAtom>(),
                                                 cute::Layout<cute::Shape<cute::Int<TileX>, cute::Int<TileY>>> {}));
    using generated = typename generated_config_getter<InputType,
                                                       OutputType,
                                                       M,
                                                       N,
                                                       K,
                                                       TransposeMode,
                                                       ThreadsAvailable,
                                                       HasBlockSize,
                                                       SM>::config;

    using config = COMMONDX_STL_NAMESPACE::conditional_t<Benchmark, manual, generated>;
    static_assert(!HasBlockSize || cute::size(config {}) <= ThreadsAvailable, "FMA CuTe tile sizes are improperly defined");
};

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_DB_HPP
