# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Affine INT4 (MLX-style) fragment-word dequant device helper.

Weights are ``w = s * q + b`` with ``q`` the raw 4-bit value in [0, 15] and a
per-group (scale ``s``, bias ``b``) pair that is NOT symmetric (``b / s`` is a
continuous float, about -7.5 on the Flint checkpoint), so the zero point cannot
be folded into the integer weights.  The cheap magic-number path of the
symmetric kernel still yields the exact integers ``q - 8`` in fp16x2 (two
``lop3``, one ``sub.f16x2``, one ``fma.rn.f16x2`` per four nibbles); the scale
multiply of the symmetric kernel becomes a single fused multiply-add::

    w = fma(q - 8, s, b')        with   b' = b + 8 * s

``b'`` is precomputed at repack time in float32 and rounded once to fp16
(``int4_affine_reference.repack_scales_biases``).  The instruction count is
unchanged versus the symmetric kernel (the ``mul.f16x2`` is replaced by an
``fma.rn.f16x2``); the only addition is one extra ``f16x2`` splat of ``b'`` per
word (``s`` is already splatted).
"""

from __future__ import annotations

from cutlass import Float16, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _dequant_int4_word_affine(q, scale, bias, *, loc=None, ip=None):
    """Dequant one 32-bit word of four raw INT4 nibbles into two ``f16x2`` words.

    ``scale`` and ``bias`` are the group's fp16 ``s`` and ``b' = b + 8 s``.  Each
    output element is ``fma.rn.f16x2((nibble - 8), s, b') = s * nibble + b``
    (single rounding).  Same fragment-order contract as the symmetric helper:
    ``dequant(word)`` / ``dequant(word >> 8)`` peel the two N-groups directly.
    """
    i32 = Int32.mlir_type
    u32 = Uint32.mlir_type
    source = Uint32(q).ir_value(loc=loc, ip=ip)
    ex = arith.constant(u32, 0x64006400, loc=loc, ip=ip)
    lut = arith.constant(u32, (0xF0 & 0xCC) | 0xAA, loc=loc, ip=ip)
    lo_mask = arith.constant(u32, 0x000F000F, loc=loc, ip=ip)
    hi_mask = arith.constant(u32, 0x00F000F0, loc=loc, ip=ip)

    def _lop3(mask):
        return llvm.inline_asm(
            i32, [source, mask, ex, lut], "lop3.b32 $0, $1, $2, $3, $4;",
            "=r,r,n,n,n", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    lo = _lop3(lo_mask)
    hi = _lop3(hi_mask)
    sub = arith.constant(u32, 0x64086408, loc=loc, ip=ip)
    lo = llvm.inline_asm(
        i32, [lo, sub], "sub.f16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    mul = arith.constant(u32, 0x2C002C00, loc=loc, ip=ip)
    add = arith.constant(u32, 0xD480D480, loc=loc, ip=ip)
    hi = llvm.inline_asm(
        i32, [hi, mul, add], "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    vec_t = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    sv = vector.broadcast(vec_t, Float16(scale).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    bv = vector.broadcast(vec_t, Float16(bias).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    lo_w = vector.fma(llvm.bitcast(vec_t, lo, loc=loc, ip=ip), sv, bv, loc=loc, ip=ip)
    hi_w = vector.fma(llvm.bitcast(vec_t, hi, loc=loc, ip=ip), sv, bv, loc=loc, ip=ip)
    return (
        Uint32(llvm.bitcast(i32, lo_w, loc=loc, ip=ip)),
        Uint32(llvm.bitcast(i32, hi_w, loc=loc, ip=ip)),
    )
