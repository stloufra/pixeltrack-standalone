// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_COMPLEX_TYPES_HPP
#define COMMONDX_COMPLEX_TYPES_HPP

#include <cuda_fp16.h>

// Namespace wrapper
#include "detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<class T>
        struct complex_base {
            using value_type = T;

            complex_base()                    = default;
            complex_base(const complex_base&) = default;
            complex_base(complex_base&&)      = default;
            __device__ __forceinline__ __host__ constexpr complex_base(value_type re, value_type im): x(re), y(im) {}

            __device__ __forceinline__ __host__ constexpr value_type real() const { return x; }
            __device__ __forceinline__ __host__ constexpr value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void                 real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void                 imag(value_type im) { y = im; }

            __device__ __forceinline__ __host__ complex_base& operator=(value_type re) {
                x = re;
                y = value_type();
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator+=(value_type re) {
                x += re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator-=(value_type re) {
                x -= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            complex_base& operator=(const complex_base&) = default;
            complex_base& operator=(complex_base&&) = default;

            template<class K>
            __device__ __forceinline__ __host__ complex_base& operator=(const complex_base<K>& other) {
                x = other.real();
                y = other.imag();
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator+=(const OtherType& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator-=(const OtherType& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator*=(const OtherType& other) {
                auto saved_x = x;
                x            = x * other.x - y * other.y;
                y            = saved_x * other.y + y * other.x;
                return *this;
            }

            /// \internal
            value_type x, y;
        };

        template<class T>
        struct complex;

        template<>
        struct alignas(2 * sizeof(__half)) complex<__half> {
            using value_type        = __half;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;

            __device__ __forceinline__ __host__ complex(const __half2& h): xy(h) {};
            __device__ __forceinline__ __host__ complex(__half re, __half im): xy(re, im) {}
            __device__ __forceinline__ __host__ complex(float re, float im):
                xy(__float2half_rn(re), __float2half_rn(im)) {}
            __device__ __forceinline__ __host__ complex(double re, double im):
                xy(__double2half(re), __double2half(im)) {}

            __device__ __forceinline__ __host__ explicit complex(const complex<float>& other);
            __device__ __forceinline__ __host__ explicit complex(const complex<double>& other);

            __device__ __forceinline__ __host__ value_type real() const { return xy.x; }
            __device__ __forceinline__ __host__ value_type imag() const { return xy.y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { xy.x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { xy.y = im; }

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                xy.x = re;
                xy.y = value_type();
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(value_type re) {
                xy.x += re;
                return *this;
            }
            __device__ __forceinline__ complex& operator-=(value_type re) {
                xy.x -= re;
                return *this;
            }
            __device__ __forceinline__ complex& operator*=(value_type re) {
                xy = __hmul2(xy, __half2 {re, re});
                return *this;
            }
            __device__ __forceinline__ complex& operator/=(value_type re) {
                xy = __h2div(xy, __half2 {re, re});
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                xy += other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                xy -= other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = xy.x;
                xy.x         = __hfma(xy.x, other.xy.x, -xy.y * other.xy.y);
                xy.y         = __hfma(saved_x, other.xy.y, xy.y * other.xy.x);
                return *this;
            }

            friend __device__ __forceinline__ complex operator*(const complex& a, const complex& b) {
                auto result = a;
                result *= b;
                return result;
            }

            friend __device__ __forceinline__ complex operator+(const complex& a, const complex& b) {
                return {a.xy + b.xy};
            }

            friend __device__ __forceinline__ complex operator-(const complex& a, const complex& b) {
                return {a.xy - b.xy};
            }

            friend __device__ __forceinline__ bool operator==(const complex& a, const complex& b) {
                return (a.xy == b.xy);
            }

            /// \internal
            __half2 xy;
        };

        template<>
        struct alignas(2 * sizeof(float)) complex<float>: complex_base<float> {
        private:
            using base_type = complex_base<float>;

        public:
            using value_type        = float;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;
            __device__ __forceinline__ __host__ constexpr complex(float re, float im): base_type(re, im) {}
            __device__ __forceinline__ __host__ explicit complex(const complex<__half>& other);
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<double>& other);

            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;
            using base_type::operator=;

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            friend __device__ __forceinline__ complex operator*(const complex& a, const complex& b) {
                return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
            }

            friend __device__ __forceinline__ complex operator+(const complex& a, const complex& b) {
                return {a.x + b.x, a.y + b.y};
            }

            friend __device__ __forceinline__ complex operator-(const complex& a, const complex& b) {
                return {a.x - b.x, a.y - b.y};
            }

            friend __device__ __forceinline__ bool operator==(const complex& a, const complex& b) {
                return (a.x == b.x) && (a.y == b.y);
            }
        };


        template<>
        struct alignas(2 * sizeof(double)) complex<double>: complex_base<double> {
        private:
            using base_type = complex_base<double>;

        public:
            using value_type        = double;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;
            __device__ __forceinline__ __host__ constexpr complex(double re, double im): base_type(re, im) {}
            __device__ __forceinline__ __host__ explicit complex(const complex<__half>& other);
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<float>& other);

            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;
            using base_type::operator=;

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            friend __device__ __host__ __forceinline__ complex operator*(const complex& a, const complex& b) {
                return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
            }

            friend __device__ __host__ __forceinline__ complex operator+(const complex& a, const complex& b) {
                return {a.x + b.x, a.y + b.y};
            }

            friend __device__ __host__ __forceinline__ complex operator-(const complex& a, const complex& b) {
                return {a.x - b.x, a.y - b.y};
            }

            friend __device__ __host__ __forceinline__ bool operator==(const complex& a, const complex& b) {
                return (a.x == b.x) && (a.y == b.y);
            }
        };

        // For FFT computations, complex<half2> should be in RRII layout.
        template<>
        struct alignas(2 * sizeof(__half2)) complex<__half2> {
            using value_type        = __half2;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;

            __device__ __forceinline__ __host__ complex(value_type re,
                                                        value_type im)
                : x(re), y(im) {}

            __device__ __forceinline__ __host__ complex(double re, double im) {
                __half hre = __double2half(re);
                x = __half2(hre, hre);
                __half him = __double2half(im);
                y = __half2(him, him);
            }

            __device__ __forceinline__ __host__ complex(float re, float im)
                : x(__float2half2_rn(re)), y(__float2half2_rn(im)) {}

            __device__ __forceinline__
                __host__ explicit complex(const complex<double> &other) {

                __half hre = __double2half(other.real());
                x = __half2(hre, hre);
                __half him = __double2half(other.imag());
                y = __half2(him, him);
            }

            __device__ __forceinline__ __host__ explicit complex(const complex<float>& other):
                x(__float2half2_rn(other.real())), y(__float2half2_rn(other.imag())) {}

            __device__ __forceinline__ __host__ value_type real() const { return x; }
            __device__ __forceinline__ __host__ value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { y = im; }

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                x = re;
                y = value_type();
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(value_type re) {
                x += re;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(value_type re) {
                x -= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = x;
                x            = __hfma2(x, other.x, -y * other.y);
                y            = __hfma2(saved_x, other.y, y * other.x);
                return *this;
            }

            /// \internal
            value_type x, y;
        };

        __host__ __device__ __forceinline__ complex<__half>::complex(const complex<float>& other):
            xy(__float2half_rn(other.real()), __float2half_rn(other.imag())) {};

        __host__ __device__ __forceinline__ complex<__half>::complex(const complex<double>& other):
            xy(__double2half(other.real()), __double2half(other.imag())) {};

        __host__ __device__ __forceinline__ complex<float>::complex(const complex<__half>& other):
            complex_base<float>(__half2float(other.real()), __half2float(other.imag())) {};

        __host__ __device__ __forceinline__ constexpr complex<float>::complex(const complex<double>& other):
            complex_base<float>(static_cast<float>(other.real()), static_cast<float>(other.imag())) {};

        __host__ __device__ __forceinline__ complex<double>::complex(const complex<__half>& other):
            complex_base<double>(__half2float(other.real()), __half2float(other.imag())) {};

        __host__ __device__ __forceinline__ constexpr complex<double>::complex(const complex<float>& other):
            complex_base<double>(other.real(), other.imag()) {};
    } // namespace detail

    template<class T>
    using complex = typename detail::complex<T>;
} // namespace commondx

// Namespace wrapper
#include "detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_COMPLEX_TYPES_HPP
