/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

//! A source of draft tokens for the chain verification of the Inkling decoder. A drafter that consumes target features (the DSpark block drafter) implements
//! observeTaps(); a pure text source can ignore them. New drafters plug in by implementing this interface and registering a type in createDraftSource().
class IDraftSource
{
public:
    virtual ~IDraftSource() = default;
    virtual char const* name() const noexcept = 0;
    //! Most tokens one draft() call returns.
    virtual int32_t maxDrafts() const noexcept = 0;
    //! Width of the target tap rows observeTaps() expects (FP32 values per position); 0 when the drafter needs no taps.
    virtual int32_t tapWidth() const noexcept = 0;
    //! Positions whose taps the drafter has seen (the next position observeTaps() expects).
    virtual int32_t contextLength() const noexcept = 0;
    virtual void beginRequest() noexcept = 0;
    //! Keeps only the first `length` positions of the context (a request that reuses a cached prefix resumes from an earlier request's rows); false when
    //! the source cannot (it then starts the request without context).
    virtual bool truncateContext(int32_t length) noexcept
    {
        (void) length;
        return false;
    }
    //! Taps [rows, tapWidth()] (device FP32) of positions pos0 .. pos0 + rows - 1; pos0 must equal contextLength().
    virtual void observeTaps(float const* taps, int32_t rows, int32_t pos0, cudaStream_t stream) = 0;
    //! Draft tokens after the anchor at position contextLength(); anchorEmbedding is the target's embedding of the anchor token (device FP16 [hidden]).
    //! The tokens land in hostTokens (pinned, maxDrafts() INT32) once the stream has finished.
    virtual void draft(void const* anchorEmbedding, int32_t anchorToken, int32_t* hostTokens, cudaStream_t stream) = 0;
};

//! Opens the drafter described by weightsFile: a sibling `<file>.json` with {"type": "..."} selects the implementation (default "inkling2_dspark_int8").
std::unique_ptr<IDraftSource> createDraftSource(
    std::filesystem::path const& weightsFile, int32_t capacity, int32_t maxAppendRows, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm
