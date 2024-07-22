// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_TYPES_HPP
#define CUBLASDX_TYPES_HPP

#include "commondx/complex_types.hpp"

namespace cublasdx {
    template<class T>
    using complex = typename commondx::complex<T>;
} // namespace cublasdx

#endif // CUBLASDX_TYPES_HPP
