#pragma once

#include <immintrin.h>

using double4 = __m256d;
using float8 = __m256;
using int8 = __m256i;
using long4 = __m256i;

using double2 = __m128d;
using float4 = __m128;
using int4 = __m128i;
using long2 = __m128i;

// Convert
template <typename T> inline T cvt(double4 a);
template <typename T> inline T cvtt(double4 a);
template <typename T> inline T cvt(float8 a);
template <typename T> inline T cvtt(float8 a);
template <typename T> inline T cvt(float4 a);
template <typename T> inline T cvtt(float4 a);
template <typename T> inline T cvt(int4 a);
template <typename T> inline T cvt(int8 a);

// AVX

// Arithmetic
inline double4 add(double4 a, double4 b) { return _mm256_add_pd(a, b); }
inline double4 operator+(double4 a, double4 b) { return add(a, b); }
inline double4& operator+=(double4& a, double4 b) { return a = add(a, b); }
inline double4 addsub(double4 a, double4 b) { return _mm256_addsub_pd(a, b); }
inline double4 div(double4 a, double4 b) { return _mm256_div_pd(a, b); }
inline double4 operator/(double4 a, double4 b) { return div(a, b); }
inline double4& operator/=(double4& a, double4 b) { return a = div(a, b); }
inline double4 hadd(double4 a, double4 b) { return _mm256_hadd_pd(a, b); }
inline double4 hsub(double4 a, double4 b) { return _mm256_hsub_pd(a, b); }
inline double4 mul(double4 a, double4 b) { return _mm256_mul_pd(a, b); }
inline double4 operator*(double4 a, double4 b) { return mul(a, b); }
inline double4& operator*=(double4& a, double4 b) { return a = mul(a, b); }
inline double4 sub(double4 a, double4 b) { return _mm256_sub_pd(a, b); }
inline double4 operator-(double4 a, double4 b) { return sub(a, b); }
inline double4& operator-=(double4& a, double4 b) { return a = sub(a, b); }

inline float8 add(float8 a, float8 b) { return _mm256_add_ps(a, b); }
inline float8 operator+(float8 a, float8 b) { return add(a, b); }
inline float8& operator+(float8& a, float8 b) { return a = add(a, b); }
inline float8 addsub(float8 a, float8 b) { return _mm256_addsub_ps(a, b); }
inline float8 div(float8 a, float8 b) { return _mm256_div_ps(a, b); }
inline float8 operator/(float8 a, float8 b) { return div(a, b); }
inline float8& operator/=(float8& a, float8 b) { return a = div(a, b); }
inline float8 hadd(float8 a, float8 b) { return _mm256_hadd_ps(a, b); }
inline float8 hsub(float8 a, float8 b) { return _mm256_hsub_ps(a, b); }
inline float8 mul(float8 a, float8 b) { return _mm256_mul_ps(a, b); }
inline float8 operator*(float8 a, float8 b) { return mul(a, b); }
inline float8& operator*=(float8& a, float8 b) { return a = mul(a, b); }
inline float8 sub(float8 a, float8 b) { return _mm256_sub_ps(a, b); }
inline float8 operator-(float8 a, float8 b) { return sub(a, b); }
inline float8& operator-=(float8& a, float8 b) { return a = sub(a, b); }

// compare
template <int imm8> inline double2 cmp(double2 a, double2 b) { return _mm_cmp_pd(a, b, imm8); }
template <int imm8> inline double4 cmp(double4 a, double4 b) { return _mm256_cmp_pd(a, b, imm8); }
template <int imm8> inline float4 cmp(float4 a, float4 b) { return _mm_cmp_ps(a, b, imm8); }
template <int imm8> inline float8 cmp(float8 a, float8 b) { return _mm256_cmp_ps(a, b, imm8); }
template <int imm8> inline double2 cmps(double2 a, double2 b) { return _mm_cmp_sd(a, b, imm8); }
template <int imm8> inline float4 cmps(float4 a, float4 b) { return _mm_cmp_ss(a, b, imm8); }

// Convert
template <> inline int4 cvt<int4>(double4 a) { return _mm256_cvtpd_epi32(a); }
template <> inline int4 cvtt<int4>(double4 a) { return _mm256_cvttpd_epi32(a); }
template <> inline float4 cvt<float4>(double4 a) { return _mm256_cvtpd_ps(a); }

template <> inline int8 cvt<int8>(float8 a) { return _mm256_cvtps_epi32(a); }
template <> inline int8 cvtt<int8>(float8 a) { return _mm256_cvttps_epi32(a); }
template <> inline double4 cvt<double4>(float4 a) { return _mm256_cvtps_pd(a); }

template <> inline double4 cvt<double4>(int4 a) { return _mm256_cvtepi32_pd(a); }
template <> inline float8 cvt<float8>(int8 a) { return _mm256_cvtepi32_ps(a); }

// Elementary Math

inline float8 rcp(float8 a) { return _mm256_rcp_ps(a); }
inline float8 rsqrt(float8 a) { return _mm256_rsqrt_ps(a); }
inline float8 sqrt(float8 a) { return _mm256_sqrt_ps(a); }
inline double4 sqrt(double4 a) { return _mm256_sqrt_pd(a); }

// Load

inline double4 broadcast(double2 const* mem_addr) { return _mm256_broadcast_pd(mem_addr); }
inline float8 broadcast(float4 const* mem_addr) { return _mm256_broadcast_ps(mem_addr); }
inline double4 broadcast(double const* mem_addr) { return _mm256_broadcast_sd(mem_addr); }
inline float8 broadcast(float const* mem_addr) { return _mm256_broadcast_ss(mem_addr); }

inline __m256i lddqu(__m256i const* mem_addr) { return _mm256_lddqu_si256(mem_addr); }

inline double4 load(double const* mem_addr) { return _mm256_load_pd(mem_addr); }
inline float8 load(float const* mem_addr) { return _mm256_load_ps(mem_addr); }
inline __m256i load(__m256i const* mem_addr) { return _mm256_load_si256(mem_addr); }
inline double4 loadu(double const* mem_addr) { return _mm256_loadu_pd(mem_addr); }
inline float8 loadu(float const* mem_addr) { return _mm256_loadu_ps(mem_addr); }
inline __m256i loadu(__m256i const* mem_addr) { return _mm256_loadu_si256(mem_addr); }

inline double4 loadu(double const* hiaddr, double const* loaddr) { return _mm256_loadu2_m128d(hiaddr, loaddr); }
inline float8 loadu(float const* hiaddr, float const* loaddr) { return _mm256_loadu2_m128(hiaddr, loaddr); }
inline __m256i loadu(__m128i const* hiaddr, __m128i const* loaddr) { return _mm256_loadu2_m128i(hiaddr, loaddr); }

inline double2 load(double const* mem_addr, long2 mask) { return _mm_maskload_pd(mem_addr, mask); }
inline double4 load(double const* mem_addr, long4 mask) { return _mm256_maskload_pd(mem_addr, mask); }
inline float4 load(float const* mem_addr, int4 mask) { return _mm_maskload_ps(mem_addr, mask); }
inline float8 load(float const* mem_addr, int8 mask) { return _mm256_maskload_ps(mem_addr, mask); }

/*
inline double4 operator+(double4 a, double4 b) { return _mm256_add_pd(a, b); }
inline double4 operator-(double4 a, double4 b) { return _mm256_sub_pd(a, b); }
*/