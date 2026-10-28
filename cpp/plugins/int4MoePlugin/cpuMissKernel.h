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

#include <immintrin.h>

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
    return t;
}

inline Tables const& tables()
{
    static Tables const t = buildTables();
    return t;
}

//! out[c] = sum_k x[k] * (q[64 * nt + c, k] - 8) * scale[64 * nt + c, k / 128] for the 64 output columns of tile nt. w: [K/16][2N] int32 of one expert,
//! s: [K/128][N] fp16 bits. Eight weights per FMA; the accumulators keep the lane order of the packing and are reduced to columns once at the end.
__attribute__((target("avx2,fma,f16c"))) inline void gemvTile(
    Tables const& t, int32_t const* w, uint16_t const* s, float const* x, int32_t K, int32_t N, int32_t nt, float* out64)
{
    constexpr int32_t kGroup = 128;
    alignas(32) float acc[128][8] = {};
    alignas(32) float part[128][8];
    alignas(32) float chunk[64];
    __m256i const mask4 = _mm256_set1_epi32(0xF);
    __m256i const eight = _mm256_set1_epi32(8);
    __m256i const seven = _mm256_set1_epi32(7);
    for (int32_t g = 0; g < K / kGroup; ++g)
    {
        for (int32_t i = 0; i < 128; ++i)
        {
            _mm256_store_ps(part[i], _mm256_setzero_ps());
        }
        for (int32_t kt = g * (kGroup / 16); kt < (g + 1) * (kGroup / 16); ++kt)
        {
            __m256 const xlo = _mm256_loadu_ps(x + kt * 16);
            __m256 const xhi = _mm256_loadu_ps(x + kt * 16 + 8);
            int32_t const* tile = w + static_cast<size_t>(kt) * 2 * N + nt * 128;
            for (int32_t b = 0; b < 16; ++b)
            {
                __m256i const v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(tile + b * 8));
#pragma GCC unroll 8
                for (int32_t j = 0; j < 8; ++j)
                {
                    __m256i const nib = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_set1_epi32(4 * j)), mask4);
                    __m256 const wf = _mm256_cvtepi32_ps(_mm256_sub_epi32(nib, eight));
                    __m256i const ix = _mm256_load_si256(reinterpret_cast<__m256i const*>(t.xIdx[b][j]));
                    __m256 const xv = _mm256_blendv_ps(_mm256_permutevar8x32_ps(xlo, ix), _mm256_permutevar8x32_ps(xhi, ix),
                        _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix, seven)));
                    float* p = part[b * 8 + j];
                    _mm256_store_ps(p, _mm256_fmadd_ps(wf, xv, _mm256_load_ps(p)));
                }
            }
        }
        uint16_t const* sc = s + static_cast<size_t>(g) * N + nt * 64;
        for (int32_t i = 0; i < 8; ++i)
        {
            _mm256_store_ps(chunk + 8 * i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(sc + 8 * i))));
        }
        for (int32_t b = 0; b < 16; ++b)
        {
            for (int32_t j = 0; j < 8; ++j)
            {
                __m256 const sv = _mm256_i32gather_ps(chunk, _mm256_load_si256(reinterpret_cast<__m256i const*>(t.sIdx[b][j])), 4);
                float* a = acc[b * 8 + j];
                _mm256_store_ps(a, _mm256_fmadd_ps(_mm256_load_ps(part[b * 8 + j]), sv, _mm256_load_ps(a)));
            }
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

} // namespace cpumiss
} // namespace plugins
} // namespace trt_edgellm
