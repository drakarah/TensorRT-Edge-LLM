/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined(__x86_64__)

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

namespace trt_edgellm
{
namespace plugins
{
namespace cpumiss
{

//! Host-side INT4 GEMV that reads the plugin's Marlin weight layout in place (see pack_int4_awq_marlin). A weight matrix [K/16][2N] int32 is cut into
//! 16 x 64 tiles of 128 uint32; slot o of a tile holds eight nibbles at fixed (row, column) positions of the tile. Scales are [K/128][N] fp16 with a fixed
//! permutation inside every 64 columns. Symmetric: value = (q - 8) * scale.
struct Tables
{
    uint8_t row[128][8];
    uint8_t col[128][8];
    int32_t inverseScalePerm[64];
    alignas(32) int32_t xIdx[16][8][8];    // [slot block][nibble][lane]: row of the tile the lane multiplies
    alignas(32) int32_t sIdx[16][8][8];    // position of the lane's column inside the stored 64-scale chunk
    alignas(32) int32_t column[16][8][8];  // output column of the tile the lane contributes to
    // The lane index vectors repeat: 8 distinct x-row vectors and 16 distinct scale-position vectors over the 128 (block, nibble) pairs
    alignas(32) int32_t xVec[8][8];
    alignas(32) int32_t sVec[16][8];
    uint8_t xOf[16][8];
    uint8_t sOf[16][8];
};

inline Tables buildTables()
{
    Tables t{};
    int32_t const rowOffset[4] = {0, 1, 8, 9};
    int32_t const pack[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    for (int32_t i = 0; i < 128; ++i)
    {
        int32_t const q = i / 4;
        int32_t const base = (q / 8) * 16 + (q % 8);
        int32_t const o = 4 * (i % 32) + i / 32;
        for (int32_t j = 0; j < 8; ++j)
        {
            int32_t const m = pack[j];
            t.row[o][j] = static_cast<uint8_t>(2 * (i % 4) + rowOffset[m % 4]);
            t.col[o][j] = static_cast<uint8_t>(base + (m >= 4 ? 8 : 0));
        }
    }
    // scale_perm[p] = i + 8 * j for p = 8 * i + j; the inverse maps an original column to its stored position
    for (int32_t i = 0; i < 8; ++i)
    {
        for (int32_t j = 0; j < 8; ++j)
        {
            t.inverseScalePerm[i + 8 * j] = 8 * i + j;
        }
    }
    for (int32_t b = 0; b < 16; ++b)
    {
        for (int32_t j = 0; j < 8; ++j)
        {
            for (int32_t l = 0; l < 8; ++l)
            {
                int32_t const o = 8 * b + l;
                t.xIdx[b][j][l] = t.row[o][j];
                t.column[b][j][l] = t.col[o][j];
                t.sIdx[b][j][l] = t.inverseScalePerm[t.col[o][j]];
            }
        }
    }
    int32_t nx = 0;
    int32_t ns = 0;
    for (int32_t b = 0; b < 16; ++b)
    {
        for (int32_t j = 0; j < 8; ++j)
        {
            int32_t d = 0;
            while (d < nx && !std::equal(t.xVec[d], t.xVec[d] + 8, t.xIdx[b][j]))
            {
                ++d;
            }
            if (d == nx)
            {
                std::copy(t.xIdx[b][j], t.xIdx[b][j] + 8, t.xVec[nx++]);
            }
            t.xOf[b][j] = static_cast<uint8_t>(d);
            int32_t e = 0;
            while (e < ns && !std::equal(t.sVec[e], t.sVec[e] + 8, t.sIdx[b][j]))
            {
                ++e;
            }
            if (e == ns)
            {
                std::copy(t.sIdx[b][j], t.sIdx[b][j] + 8, t.sVec[ns++]);
            }
            t.sOf[b][j] = static_cast<uint8_t>(e);
        }
    }
    return t;
}

inline Tables const& tables()
{
    static Tables const t = buildTables();
    return t;
}

//! One 8-lane block b of a group for gemvTile. PAR = b & 1 selects the x vectors: the distinct x-row vector of nibble j is 4 * PAR + kXOfNibble[j],
//! which makes the xp offsets compile-time constants (checked against Tables::xOf by marlin_kernel_bench).
inline constexpr int32_t kXOfNibble[8] = {0, 1, 0, 1, 2, 3, 2, 3};

template <int32_t PAR>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline void blockFast(Tables const& t, int32_t const* tile0, int32_t N, int32_t b, float const (*xp)[8][8],
    float const (*sx)[8], float const (*sv)[8], __m256 const* shiftBack, __m256i const* m, float (*acc)[8])
{
    constexpr int32_t kTiles = 8;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int32_t k = 0; k < kTiles; ++k)
    {
        __m256i const v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(tile0 + static_cast<size_t>(k) * 2 * N + b * 8));
        float const(*xk)[8] = xp[k] + PAR * 4;
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[0])), _mm256_load_ps(xk[kXOfNibble[0]]), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[1])), _mm256_load_ps(xk[kXOfNibble[1]]), a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[2])), _mm256_load_ps(xk[kXOfNibble[2]]), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[3])), _mm256_load_ps(xk[kXOfNibble[3]]), a3);
        a4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[4])), _mm256_load_ps(xk[kXOfNibble[4]]), a4);
        a5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[5])), _mm256_load_ps(xk[kXOfNibble[5]]), a5);
        a6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m[6])), _mm256_load_ps(xk[kXOfNibble[6]]), a6);
        a7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 28)), _mm256_load_ps(xk[kXOfNibble[7]]), a7);
    }
    __m256 const a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int32_t j = 0; j < 8; ++j)
    {
        __m256 const centred = _mm256_fmsub_ps(a[j], shiftBack[j], _mm256_load_ps(sx[PAR * 4 + kXOfNibble[j]]));
        float* p = acc[b * 8 + j];
        _mm256_store_ps(p, _mm256_fmadd_ps(centred, _mm256_load_ps(sv[t.sOf[b][j]]), _mm256_load_ps(p)));
    }
}

//! out[c] = sum_k x[k] * (q[64 * nt + c, k] - 8) * scale[64 * nt + c, k / 128] for the 64 output columns of tile nt. w: [K/16][2N] int32 of one expert,
//! s: [K/128][N] fp16 bits. The accumulators keep the lane order of the packing and are reduced to columns once at the end. The lane index vectors that gather x and the scales repeat (8 and 16 distinct), so
//! they are built once per 128-wide group; the weight offset (q - 8) is applied as one correction of 8 * sum(x) per group; 8 accumulators stay in
//! registers across the 8 k-tiles of a group, so the scale multiply and the memory update happen once per group.
__attribute__((target("avx2,fma,f16c"))) inline void gemvTile(
    Tables const& t, int32_t const* w, uint16_t const* s, float const* x, int32_t K, int32_t N, int32_t nt, float* out64)
{
    constexpr int32_t kGroup = 128;
    constexpr int32_t kTiles = kGroup / 16;
    alignas(32) float acc[128][8] = {};
    alignas(32) float xp[kTiles][8][8];
    alignas(32) float sx[8][8];
    alignas(32) float sv[16][8];
    alignas(32) float chunk[64];
    __m256i const seven = _mm256_set1_epi32(7);
    __m256 const eight = _mm256_set1_ps(8.0F);
    __m256i m[7];
    for (int32_t j = 0; j < 7; ++j)
    {
        m[j] = _mm256_set1_epi32(0xF << (4 * j));
    }
    __m256 shiftBack[8];
    for (int32_t j = 0; j < 8; ++j)
    {
        // nibble 7 is shifted down (its in-place value would overflow int32), so it needs no correction
        shiftBack[j] = _mm256_set1_ps(j == 7 ? 1.0F : 1.0F / static_cast<float>(1u << (4 * j)));
    }
    for (int32_t g = 0; g < K / kGroup; ++g)
    {
        __m256 sum[8];
        for (int32_t d = 0; d < 8; ++d)
        {
            sum[d] = _mm256_setzero_ps();
        }
        for (int32_t k = 0; k < kTiles; ++k)
        {
            __m256 const xlo = _mm256_loadu_ps(x + (g * kTiles + k) * 16);
            __m256 const xhi = _mm256_loadu_ps(x + (g * kTiles + k) * 16 + 8);
            for (int32_t d = 0; d < 8; ++d)
            {
                __m256i const ix = _mm256_load_si256(reinterpret_cast<__m256i const*>(t.xVec[d]));
                __m256 const xv = _mm256_blendv_ps(_mm256_permutevar8x32_ps(xlo, ix), _mm256_permutevar8x32_ps(xhi, ix), _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix, seven)));
                _mm256_store_ps(xp[k][d], xv);
                sum[d] = _mm256_add_ps(sum[d], xv);
            }
        }
        for (int32_t d = 0; d < 8; ++d)
        {
            _mm256_store_ps(sx[d], _mm256_mul_ps(sum[d], eight));
        }
        uint16_t const* sc = s + static_cast<size_t>(g) * N + nt * 64;
        for (int32_t i = 0; i < 8; ++i)
        {
            _mm256_store_ps(chunk + 8 * i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(sc + 8 * i))));
        }
        for (int32_t e = 0; e < 16; ++e)
        {
            _mm256_store_ps(sv[e], _mm256_i32gather_ps(chunk, _mm256_load_si256(reinterpret_cast<__m256i const*>(t.sVec[e])), 4));
        }
        int32_t const* tile0 = w + static_cast<size_t>(g) * kTiles * 2 * N + nt * 128;
        for (int32_t b = 0; b < 16; b += 2)
        {
            blockFast<0>(t, tile0, N, b, xp, sx, sv, shiftBack, m, acc);
            blockFast<1>(t, tile0, N, b + 1, xp, sx, sv, shiftBack, m, acc);
        }
    }
    for (int32_t c = 0; c < 64; ++c)
    {
        out64[c] = 0.0F;
    }
    for (int32_t b = 0; b < 16; ++b)
    {
        for (int32_t j = 0; j < 8; ++j)
        {
            for (int32_t l = 0; l < 8; ++l)
            {
                out64[t.column[b][j][l]] += acc[b * 8 + j][l];
            }
        }
    }
}

//! int8 activations of one input vector for gemvTileVnni: symmetric, one scale per 128 inputs, laid out per 16-row k-tile and block parity as 8 lanes x 4 bytes (rows
//! 2p + {0, 8, 1, 9} of the tile, p = (2b + lane / 4) % 4 for blocks b of the tile), plus per group the int32 sum of every lane's four quantised values.
template <int32_t kMaxK>
struct QuantisedX
{
    alignas(32) int8_t v[kMaxK / 16 * 64];
    alignas(32) int32_t sum[kMaxK / 128 * 16];
    float scale[kMaxK / 128];
    int32_t k;
};

template <int32_t kMaxK>
__attribute__((target("avx2,fma,f16c,avxvnni"))) inline void quantiseX(float const* x, int32_t K, QuantisedX<kMaxK>& q)
{
    static __m128i const* const masks = []()
    {
        alignas(16) static uint8_t bytes[4][16];
        for (int32_t p = 0; p < 4; ++p)
        {
            for (int32_t l = 0; l < 4; ++l)
            {
                bytes[p][4 * l + 0] = static_cast<uint8_t>(2 * p);
                bytes[p][4 * l + 1] = static_cast<uint8_t>(2 * p + 8);
                bytes[p][4 * l + 2] = static_cast<uint8_t>(2 * p + 1);
                bytes[p][4 * l + 3] = static_cast<uint8_t>(2 * p + 9);
            }
        }
        return reinterpret_cast<__m128i const*>(bytes);
    }();
    q.k = K;
    __m256i const ones = _mm256_set1_epi8(1);
    __m256 const signMask = _mm256_set1_ps(-0.0F);
    for (int32_t g = 0; g < K / 128; ++g)
    {
        float const* xg = x + g * 128;
        __m256 amaxv = _mm256_setzero_ps();
        for (int32_t i = 0; i < 128; i += 8)
        {
            amaxv = _mm256_max_ps(amaxv, _mm256_andnot_ps(signMask, _mm256_loadu_ps(xg + i)));
        }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(amaxv), _mm256_extractf128_ps(amaxv, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float const amax = _mm_cvtss_f32(m4);
        float const scale = amax > 0.0F ? amax / 127.0F : 1.0F;
        __m256 const inv = _mm256_set1_ps(1.0F / scale);
        q.scale[g] = scale;
        for (int32_t k = 0; k < 8; ++k)
        {
            __m256i const a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + k * 16), inv));
            __m256i const b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + k * 16 + 8), inv));
            __m256i const words = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
            __m256i const bytes = _mm256_packs_epi16(words, words);
            __m128i const row = _mm_unpacklo_epi64(_mm256_castsi256_si128(bytes), _mm256_extracti128_si256(bytes, 1));
            int8_t* dst = q.v + static_cast<size_t>(g * 8 + k) * 64;
            for (int32_t par = 0; par < 2; ++par)
            {
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + par * 32), _mm_shuffle_epi8(row, _mm_load_si128(masks + 2 * par)));
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + par * 32 + 16), _mm_shuffle_epi8(row, _mm_load_si128(masks + 2 * par + 1)));
            }
        }
        for (int32_t par = 0; par < 2; ++par)
        {
            __m256i s = _mm256_setzero_si256();
            for (int32_t k = 0; k < 8; ++k)
            {
                s = _mm256_dpbusd_avx_epi32(s, ones, _mm256_load_si256(reinterpret_cast<__m256i const*>(q.v + static_cast<size_t>(g * 8 + k) * 64 + par * 32)));
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(q.sum + (g * 2 + par) * 8), s);
        }
    }
}

//! gemvTile with int8 activations: u8 weights x s8 activations with vpdpbusd; per group the integer sum minus 8 * sum(x) is scaled by the weight scale times the
//! activation scale. Nibbles {0,1,4,5} of a word are one output column and {2,3,6,7} the column 8 away; both multiply the same four activation rows.
template <int32_t kMaxK>
__attribute__((target("avx2,fma,f16c,avxvnni"))) inline void gemvTileVnni(
    Tables const& t, int32_t const* w, uint16_t const* s, QuantisedX<kMaxK> const& x, int32_t N, int32_t nt, float* out64)
{
    alignas(32) float accF[16][2][8] = {};
    alignas(32) float sv[16][8];
    alignas(32) float chunk[64];
    __m256i const m0 = _mm256_set1_epi32(0x000F000F);
    __m256i const m1 = _mm256_set1_epi32(0x0F000F00);
    __m256i const eight = _mm256_set1_epi32(8);
    for (int32_t g = 0; g < x.k / 128; ++g)
    {
        uint16_t const* sc = s + static_cast<size_t>(g) * N + nt * 64;
        for (int32_t i = 0; i < 8; ++i)
        {
            _mm256_store_ps(chunk + 8 * i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(sc + 8 * i))));
        }
        __m256 const xs = _mm256_set1_ps(x.scale[g]);
        for (int32_t e = 0; e < 16; ++e)
        {
            _mm256_store_ps(sv[e], _mm256_mul_ps(_mm256_i32gather_ps(chunk, _mm256_load_si256(reinterpret_cast<__m256i const*>(t.sVec[e])), 4), xs));
        }
        __m256i const off[2] = {_mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<__m256i const*>(x.sum + (g * 2 + 0) * 8)), eight),
            _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<__m256i const*>(x.sum + (g * 2 + 1) * 8)), eight)};
        int32_t const* tile0 = w + static_cast<size_t>(g) * 8 * 2 * N + nt * 128;
        for (int32_t b = 0; b < 16; b += 4)
        {
            __m256i accA[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            __m256i accB[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            for (int32_t k = 0; k < 8; ++k)
            {
                int32_t const* tk = tile0 + static_cast<size_t>(k) * 2 * N + b * 8;
                int8_t const* xk = x.v + static_cast<size_t>(g * 8 + k) * 64;
                __m256i const xv[2] = {_mm256_load_si256(reinterpret_cast<__m256i const*>(xk)), _mm256_load_si256(reinterpret_cast<__m256i const*>(xk + 32))};
#pragma GCC unroll 4
                for (int32_t i = 0; i < 4; ++i)
                {
                    __m256i const v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(tk + i * 8));
                    __m256i const lo = _mm256_or_si256(_mm256_and_si256(v, m0), _mm256_and_si256(_mm256_slli_epi32(v, 4), m1));
                    __m256i const hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(v, 8), m0), _mm256_and_si256(_mm256_srli_epi32(v, 4), m1));
                    accA[i] = _mm256_dpbusd_avx_epi32(accA[i], lo, xv[i & 1]);
                    accB[i] = _mm256_dpbusd_avx_epi32(accB[i], hi, xv[i & 1]);
                }
            }
            for (int32_t i = 0; i < 4; ++i)
            {
                int32_t const bb = b + i;
                __m256 const ca = _mm256_cvtepi32_ps(_mm256_sub_epi32(accA[i], off[bb & 1]));
                __m256 const cb = _mm256_cvtepi32_ps(_mm256_sub_epi32(accB[i], off[bb & 1]));
                _mm256_store_ps(accF[bb][0], _mm256_fmadd_ps(ca, _mm256_load_ps(sv[t.sOf[bb][0]]), _mm256_load_ps(accF[bb][0])));
                _mm256_store_ps(accF[bb][1], _mm256_fmadd_ps(cb, _mm256_load_ps(sv[t.sOf[bb][2]]), _mm256_load_ps(accF[bb][1])));
            }
        }
    }
    for (int32_t c = 0; c < 64; ++c)
    {
        out64[c] = 0.0F;
    }
    for (int32_t b = 0; b < 16; ++b)
    {
        for (int32_t l = 0; l < 8; ++l)
        {
            out64[t.column[b][0][l]] += accF[b][0][l];
            out64[t.column[b][2][l]] += accF[b][1][l];
        }
    }
}

} // namespace cpumiss
} // namespace plugins
} // namespace trt_edgellm

#endif // defined(__x86_64__)
