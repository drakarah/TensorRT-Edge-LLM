"""Warp-specialized head-512 causal context FMHA for Ampere-class GPUs (CuTe DSL).

Same ABI as ``FMHAV2Ampere.__call_context__`` (dense BSND Q/K/V/O, logical
``cum_seqlen_k``, bottom-right causal), specialised for ``head_dim == 512``.

Why a separate kernel: the generic FMHA-v2 kernel splits D512 into two CTAs
that each own one 256-column output half, so every CTA recomputes Q·Kᵀ and
streams the full K tile from L2 for a 32-row query tile (Nsight Compute on
sm_87: tensor pipe 47 %, L2 86 %, 4 warps per SM).

This kernel keeps one CTA per 32-row query tile with 8 warps in two roles:

* S warps (0-3): two groups of two warps; group ``kh`` computes S = Q·Kᵀ for
  the 16-key half ``kh`` of each 32-key tile.  Q lives in registers for the
  whole CTA (one ldmatrix pass), so only K is read from shared memory per
  tile.  The groups exchange per-row maxima through shared memory, run the
  online softmax, and publish P (fp16) plus the per-row rescale factor.
* O warps (4-7): two groups of two warps; group ``c`` owns the 256-column
  output half ``c`` (fp32 accumulator in registers) and computes O += P·V.

P and the rescale factor are double-buffered in shared memory; K and V tiles
are double-buffered and loaded by the role that consumes them.  Q·Kᵀ is
computed once per tile and K is read from L2 once per query tile.  The MMA
work per tile is the same on each of the four SM sub-partitions (warp ``w``
issues on scheduler ``w % 4``: one S warp and one O warp each).

Named barriers (ids):
  1-2  S group ``kh`` (64 threads): K stage visible / free
  3-4  O group ``c`` (64 threads): V stage visible / free
  5    all S warps (128): Q visible, row-max exchange
  6-7  P_FULL[stage]  (256): S arrive, O sync
  8-9  P_EMPTY[stage] (256): O arrive, S sync
  10   FINAL (256): S arrive with the row sums, O sync
  11   QFREE (256): S arrive once Q is in registers, O sync before reusing
       the Q buffer (which aliases V stage 1)
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp

_BR = 32  # query rows per CTA
_BC = 32  # keys per tile
_D = 512
_DO = 256  # output columns per O group
_THREADS = 256

_BAR_S_GRP = 1
_BAR_O_GRP = 3
_BAR_S_ALL = 5
_BAR_P_FULL = 6
_BAR_P_EMPTY = 8
_BAR_FINAL = 10
_BAR_QFREE = 11


class FMHAV2D512WS:
    def __init__(self, is_causal: bool = True, skip_rescale: bool = True, num_acc_s: int = 4,
                 head_dim: int = 512):
        # head_dim 512 (Gemma 4 global layers) or 256 (Qwen3.5 full attention, Gemma 4 sliding layers):
        # each O group owns head_dim / 2 output columns.
        if head_dim not in (256, 512):
            raise ValueError("head_dim must be 256 or 512")
        self._D = head_dim
        self._DO = head_dim // 2
        if not is_causal:
            raise ValueError("FMHAV2D512WS implements the causal context variant only")
        self._skip_rescale = skip_rescale
        # Independent Q·Kᵀ accumulators (split over the 32 k-steps) so each
        # S warp keeps several HMMA chains in flight.
        self._num_acc_s = num_acc_s
        self._dtype = cutlass.Float16

    @cute.jit
    def __call_context__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        cum_seqlen_k: cute.Tensor,
        block_begin,
        block_end,
        window_size_left: cutlass.Int32,
        attention_scale: cutlass.Float32,
        scale_q: cutlass.Float32,
        scale_k: cutlass.Float32,
        scale_v: cutlass.Float32,
        inv_scale_o: cutlass.Float32,
        sm_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(block_begin is not None or block_end is not None):
            raise TypeError("FMHAV2D512WS has no vision-block variant")
        if cutlass.const_expr(q_tensor.element_type != cutlass.Float16):
            raise TypeError("FMHAV2D512WS supports Float16 only")

        atom64 = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        atom32 = cute.make_composed_layout(
            cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1))
        )
        sQ_layout = cute.tile_to_shape(atom64, (_BR, self._D), (0, 1))
        # K: (16 keys, D, stage * 2 + key half)
        sK_layout = cute.tile_to_shape(atom64, (_BC // 2, self._D, 4), (0, 1, 2))
        # V: (32 keys, 256 columns, stage * 2 + column half)
        sV_layout = cute.tile_to_shape(atom64, (_BC, self._DO, 4), (0, 1, 2))
        # P: (32 rows, 32 keys, stage)
        sP_layout = cute.tile_to_shape(atom32, (_BR, _BC, 2), (0, 1, 2))

        @cute.struct
        class SharedStorage:
            sV: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(sV_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(sK_layout)], 1024
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(sP_layout)], 1024
            ]
            sMax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BR * 4], 16]
            sAlpha: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BR * 2], 16]
            sRowSum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BR * 2], 16]

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float16,
            num_bits_per_copy=128,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128
        )
        v_layout = cute.make_layout((1, 8))
        copy64 = cute.make_tiled_copy_tv(
            atom_async_copy, cute.make_layout((8, 8), stride=(8, 1)), v_layout
        )
        copy128 = cute.make_tiled_copy_tv(
            atom_async_copy, cute.make_layout((16, 8), stride=(8, 1)), v_layout
        )
        copy_o = cute.make_tiled_copy_tv(
            atom_universal_copy, cute.make_layout((8, 8), stride=(8, 1)), v_layout
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            (2, 1, 1),
            permutation_mnk=(32, 16, 16),
        )

        log2_e = 1.4426950408889634074
        softmax_scale_log2 = attention_scale * scale_q * scale_k * log2_e
        _ = scale_v * inv_scale_o
        _ = sm_count
        _ = window_size_left
        num_kv_heads = k_tensor.shape[2]
        grid_dim = (
            cute.ceil_div(q_tensor.shape[1], _BR),
            cute.size(q_tensor.shape[2]),
            cute.size(q_tensor.shape[0]),
        )
        self.kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            cum_seqlen_k,
            softmax_scale_log2,
            num_kv_heads,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            copy64,
            copy128,
            copy_o,
            tiled_mma,
            SharedStorage,
        ).launch(grid=grid_dim, block=[_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqLenK: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        num_kv_heads: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        copy64: cute.TiledCopy,
        copy128: cute.TiledCopy,
        copy_o: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        grp = (warp_idx // 2) % 2
        gtid = tidx % 64

        kv_group_size = mQ.shape[2] // num_kv_heads
        # Launch order: the query heads of one KV head are adjacent, so CTAs
        # that stream the same K/V tiles run together (L2 reuse); query tiles
        # go longest (last) first to shorten the causal tail.
        num_m_blocks = cute.ceil_div(mQ.shape[1], _BR)
        linear = by * num_m_blocks + bx
        rest = linear // kv_group_size
        head_kv = rest // num_m_blocks
        head = head_kv * kv_group_size + linear % kv_group_size
        m_block = num_m_blocks - 1 - rest % num_m_blocks
        seqlen_q = mQ.shape[1]
        seqlen_k = mCuSeqLenK[batch + 1] - mCuSeqLenK[batch]
        offset = seqlen_k - seqlen_q
        n_block_max = cutlass.min(
            cute.ceil_div((m_block + 1) * _BR + offset, _BC),
            cute.ceil_div(seqlen_k, _BC),
        )
        # Whole-padding query tile: no KV tiles, the epilogue stores zeros.
        num_tiles = 0 if m_block * _BR >= seqlen_q else n_block_max

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sV = storage.sV.get_tensor(sV_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sP = storage.sP.get_tensor(sP_layout)
        # Q aliases V stage 1 (blocks 2 and 3); it is read once into registers.
        sQ = cute.make_tensor(sV.iterator + 2 * _BC * self._DO, sQ_layout)
        sMax = storage.sMax.get_tensor(cute.make_layout((_BR, 2, 2)))
        sAlpha = storage.sAlpha.get_tensor(cute.make_layout((_BR, 2)))
        sRowSum = storage.sRowSum.get_tensor(cute.make_layout((_BR, 2)))

        thr_mma = tiled_mma.get_slice(gtid)
        if warp_idx < 4:
            self._s_role(
                mQ, mK, sQ, sK, sP, sMax, sAlpha, sRowSum, copy64, copy128, tiled_mma,
                thr_mma, tidx, gtid, grp, m_block, head, head_kv, batch, seqlen_q,
                seqlen_k, offset, num_tiles, softmax_scale_log2,
            )
        else:
            self._o_role(
                mV, mO, sV, sP, sAlpha, sRowSum, copy64, copy_o, tiled_mma, thr_mma,
                gtid, grp, m_block, head, head_kv, batch, seqlen_q, seqlen_k, num_tiles,
            )

    # ------------------------------------------------------------------ S role
    @cute.jit
    def _s_role(
        self, mQ, mK, sQ, sK, sP, sMax, sAlpha, sRowSum, copy64, copy128, tiled_mma,
        thr_mma, tidx, gtid, grp, m_block, head, head_kv, batch, seqlen_q, seqlen_k,
        offset, num_tiles, softmax_scale_log2,
    ):
        # Q: all 128 S threads, zero-fill rows past the sequence.
        gQ = cute.local_tile(mQ[batch, None, head, None], (_BR, self._D), (m_block, 0))
        thr_q = copy128.get_slice(tidx)
        tQgQ = thr_q.partition_S(gQ)
        tQsQ = thr_q.partition_D(sQ)
        cQ = cute.local_tile(
            cute.make_identity_tensor((mQ.shape[1], self._D)), (_BR, self._D), (m_block, 0)
        )
        tQcQ = thr_q.partition_S(cQ)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], seqlen_q):
                cute.copy(copy128, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)

        # K: this group's 16-key half of each tile.
        gK = cute.local_tile(mK[batch, None, head_kv, None], (_BC // 2, self._D), (None, 0))
        cK = cute.local_tile(
            cute.make_identity_tensor((mK.shape[1], self._D)), (_BC // 2, self._D), (None, 0)
        )
        thr_k = copy64.get_slice(gtid)
        tKgK = thr_k.partition_S(gK)
        tKcK = thr_k.partition_S(cK)
        tKsK = thr_k.partition_D(sK)

        if num_tiles > 0:
            self._load_k(copy64, tKgK, tKcK, tKsK, num_tiles, 0, 0, grp, seqlen_k, True)
        cute.arch.cp_async_commit_group()
        if num_tiles > 1:
            self._load_k(copy64, tKgK, tKcK, tKsK, num_tiles, 1, 1, grp, seqlen_k, False)
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier(barrier_id=_BAR_S_ALL, number_of_threads=128)

        # Q fragments for the whole CTA lifetime.
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        smem_copy_q = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16),
            tiled_mma,
        )
        thr_sq = smem_copy_q.get_slice(gtid)
        cute.copy(smem_copy_q, thr_sq.partition_S(sQ), thr_sq.retile(tSrQ))

        smem_copy_k = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16),
            tiled_mma,
        )
        thr_sk = smem_copy_k.get_slice(gtid)
        tSsK = thr_sk.partition_S(sK)
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK[None, None, 0]))
        tSrK_view = thr_sk.retile(tSrK)

        # Local (row, key) coordinates of the S accumulator.
        tScS = thr_mma.partition_C(cute.make_identity_tensor((_BR, _BC // 2)))
        tScS_mn = self._mn_view(tScS)
        acc_shape_s = thr_mma.partition_shape_C((_BR, _BC // 2))
        row_max = cute.make_rmem_tensor((cute.size(tScS_mn, mode=[0]),), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((cute.size(tScS_mn, mode=[0]),), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        st = (
            tSrQ, tSsK, tSrK, tSrK_view, smem_copy_k, tiled_mma, acc_shape_s, tScS,
            tScS_mn, row_max, row_sum, copy64, tKgK, tKcK, tKsK, sP, sMax, sAlpha,
        )
        if num_tiles == 0:
            cute.arch.barrier_arrive(barrier_id=_BAR_QFREE, number_of_threads=_THREADS)
        if num_tiles > 0:
            self._s_step(st, 0, 0, grp, m_block, num_tiles, seqlen_q, seqlen_k, offset,
                         softmax_scale_log2, True, True)
        if num_tiles > 1:
            self._s_step(st, 1, 1, grp, m_block, num_tiles, seqlen_q, seqlen_k, offset,
                         softmax_scale_log2, True, False)
        for i in range(2, num_tiles, 1):
            self._s_step(st, i, i % 2, grp, m_block, num_tiles, seqlen_q, seqlen_k, offset,
                         softmax_scale_log2, False, False)

        lane = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            rs = self._quad_sum(row_sum[r])
            if lane % 4 == 0:
                sRowSum[tScS_mn[r, 0][0], grp] = rs
        cute.arch.barrier_arrive(barrier_id=_BAR_FINAL, number_of_threads=_THREADS)

    @cute.jit
    def _load_k(self, copy64, tKgK, tKcK, tKsK, num_tiles, i, s, grp, seqlen_k,
                first: cutlass.Constexpr):
        nb = (num_tiles - 1 - i) * 2 + grp
        idx = s * 2 + grp
        if cutlass.const_expr(first):
            for m in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKcK[0, m, 0, nb][0], seqlen_k):
                    cute.copy(copy64, tKgK[None, m, None, nb], tKsK[None, m, None, idx])
                else:
                    tKsK[None, m, None, idx].fill(0)
        else:
            cute.copy(copy64, tKgK[None, None, None, nb], tKsK[None, None, None, idx])

    @cute.jit
    def _s_step(self, st, i, s, grp, m_block, num_tiles, seqlen_q, seqlen_k, offset,
                softmax_scale_log2, masked: cutlass.Constexpr, first: cutlass.Constexpr):
        (tSrQ, tSsK, tSrK, tSrK_view, smem_copy_k, tiled_mma, acc_shape_s, tScS,
         tScS_mn, row_max, row_sum, copy64, tKgK, tKcK, tKsK, sP, sMax, sAlpha) = st
        idx = s * 2 + grp
        n_block = num_tiles - 1 - i

        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier(barrier_id=_BAR_S_GRP + grp, number_of_threads=64)

        # S = Q·Kᵀ over the 32 k-steps, round-robin over independent accumulators.
        accs = []
        for a in cutlass.range_constexpr(self._num_acc_s):
            acc = cute.make_rmem_tensor(acc_shape_s, cutlass.Float32)
            acc.fill(0.0)
            accs.append(acc)
        num_k = cute.size(tSsK.shape[2])
        cute.copy(smem_copy_k, tSsK[None, None, 0, idx], tSrK_view[None, None, 0])
        for k in cutlass.range_constexpr(num_k):
            if cutlass.const_expr(k + 1 < num_k):
                cute.copy(smem_copy_k, tSsK[None, None, k + 1, idx], tSrK_view[None, None, k + 1])
            acc = accs[k % self._num_acc_s]
            cute.gemm(tiled_mma, acc, tSrQ[None, None, k], tSrK[None, None, k], acc)
        acc_s = accs[0]
        for a in cutlass.range_constexpr(1, self._num_acc_s):
            acc_s.store(acc_s.load() + accs[a].load())

        if cutlass.const_expr(first):
            cute.arch.barrier_arrive(barrier_id=_BAR_QFREE, number_of_threads=_THREADS)

        # K stage free: prefetch tile i + 2 into it.
        cute.arch.barrier(barrier_id=_BAR_S_GRP + grp, number_of_threads=64)
        if i + 2 < num_tiles:
            self._load_k(copy64, tKgK, tKcK, tKsK, num_tiles, i + 2, s, grp, seqlen_k, False)
        cute.arch.cp_async_commit_group()

        acc_mn = self._mn_view(acc_s)
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(masked):
            for r in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                row = m_block * _BR + tScS_mn[r, 0][0]
                col_limit = cutlass.min(row + 1 + offset, seqlen_k)
                col_limit = 0 if row + 1 > seqlen_q else col_limit
                for c in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                    col = n_block * _BC + grp * (_BC // 2) + tScS_mn[0, c][1]
                    if cute.elem_less(col_limit, col + 1):
                        acc_mn[r, c] = -cutlass.Float32.inf

        # Row-max exchange between the two key halves.
        local_max = cute.make_rmem_tensor((cute.size(row_max),), cutlass.Float32)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            m = acc_mn[r, None].load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            m = self._quad_max(m)
            local_max[r] = m
            if lane % 4 == 0:
                sMax[tScS_mn[r, 0][0], grp, s] = m
        cute.arch.barrier(barrier_id=_BAR_S_ALL, number_of_threads=128)

        alpha = cute.make_rmem_tensor((cute.size(row_max),), cutlass.Float32)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            other = sMax[tScS_mn[r, 0][0], 1 - grp, s]
            prev = row_max[r]
            cur = cute.arch.fmax(prev, cute.arch.fmax(local_max[r], other))
            safe = 0.0 if cur == -cutlass.Float32.inf else cur
            row_exp = cute.math.exp2(
                acc_mn[r, None].load() * softmax_scale_log2 - safe * softmax_scale_log2,
                fastmath=True,
            )
            rsum = row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            a = cute.math.exp2(
                prev * softmax_scale_log2 - safe * softmax_scale_log2, fastmath=True
            )
            if cutlass.const_expr(self._skip_rescale):
                a = cutlass.Float32(1.0) if a > cutlass.Float32(0.99609375) else a
            row_sum[r] = rsum + row_sum[r] * a
            row_max[r] = cur
            alpha[r] = a
            acc_mn[r, None] = row_exp

        rP = cute.make_fragment_like(acc_s, cutlass.Float16)
        rP.store(acc_s.load().to(cutlass.Float16))

        if i >= 2:
            cute.arch.barrier(barrier_id=_BAR_P_EMPTY + s, number_of_threads=_THREADS)
        for e in cutlass.range_constexpr(cute.size(rP)):
            coord = tScS[e]
            sP[coord[0], grp * (_BC // 2) + coord[1], s] = rP[e]
        if grp == 0:
            if lane % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(row_max)):
                    sAlpha[tScS_mn[r, 0][0], s] = alpha[r]
        cute.arch.barrier_arrive(barrier_id=_BAR_P_FULL + s, number_of_threads=_THREADS)

    # ------------------------------------------------------------------ O role
    @cute.jit
    def _o_role(
        self, mV, mO, sV, sP, sAlpha, sRowSum, copy64, copy_o, tiled_mma, thr_mma,
        gtid, grp, m_block, head, head_kv, batch, seqlen_q, seqlen_k, num_tiles,
    ):
        gV = cute.local_tile(mV[batch, None, head_kv, None], (_BC, self._DO), (None, grp))
        cV = cute.local_tile(
            cute.make_identity_tensor((mV.shape[1], self._D)), (_BC, self._DO), (None, grp)
        )
        thr_v = copy64.get_slice(gtid)
        tVgV = thr_v.partition_S(gV)
        tVcV = thr_v.partition_S(cV)
        tVsV = thr_v.partition_D(sV)

        if num_tiles > 0:
            self._load_v(copy64, tVgV, tVcV, tVsV, num_tiles, 0, 0, grp, seqlen_k, True)
        cute.arch.cp_async_commit_group()
        cute.arch.barrier(barrier_id=_BAR_QFREE, number_of_threads=_THREADS)
        if num_tiles > 1:
            self._load_v(copy64, tVgV, tVcV, tVsV, num_tiles, 1, 1, grp, seqlen_k, False)
        cute.arch.cp_async_commit_group()

        # Transposed view of V for the P·V B operand: (256 columns, 32 keys, block).
        sVt = cute.composition(
            sV, cute.make_layout((self._DO, _BC, 4), stride=(_BC, 1, _BC * self._DO))
        )
        smem_copy_v = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.Float16),
            tiled_mma,
        )
        thr_sv = smem_copy_v.get_slice(gtid)
        tOsVt = thr_sv.partition_S(sVt)
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt[None, None, 0]))
        tOrVt_view = thr_sv.retile(tOrVt)

        smem_copy_p = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16),
            tiled_mma,
        )
        thr_sp = smem_copy_p.get_slice(gtid)
        tOsP = thr_sp.partition_S(sP)
        tOrP = thr_mma.make_fragment_A(thr_mma.partition_A(sP[None, None, 0]))
        tOrP_view = thr_sp.retile(tOrP)

        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((_BR, self._DO)), cutlass.Float32)
        acc_O.fill(0.0)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor((_BR, self._DO)))
        tOcO_mn = self._mn_view(tOcO)

        st = (tOsVt, tOrVt, tOrVt_view, smem_copy_v, tOsP, tOrP, tOrP_view, smem_copy_p,
              tiled_mma, acc_O, tOcO_mn, sAlpha, copy64, tVgV, tVcV, tVsV)
        if num_tiles > 0:
            self._o_step(st, 0, 0, grp, num_tiles, seqlen_k, True)
        if num_tiles > 1:
            self._o_step(st, 1, 1, grp, num_tiles, seqlen_k, False)
        for i in range(2, num_tiles, 1):
            self._o_step(st, i, i % 2, grp, num_tiles, seqlen_k, False)

        # Epilogue: normalise, registers → smem (V stage 0, own block) → gmem.
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=_BAR_FINAL, number_of_threads=_THREADS)
        acc_mn = self._mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
            row = tOcO_mn[r, 0][0]
            rs = sRowSum[row, 0] + sRowSum[row, 1]
            bad = rs == 0.0 or rs != rs
            scale = 1.0 if bad else cute.arch.rcp_approx(rs)
            acc_mn[r, None] = acc_mn[r, None].load() * scale
        rO = cute.make_fragment_like(acc_O, cutlass.Float16)
        rO.store(acc_O.load().to(cutlass.Float16))

        sO = sV[None, None, grp]
        smem_copy_o = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16), tiled_mma
        )
        thr_so = smem_copy_o.get_slice(gtid)
        cute.copy(smem_copy_o, thr_so.retile(rO), thr_so.partition_D(sO))
        cute.arch.barrier(barrier_id=_BAR_O_GRP + grp, number_of_threads=64)

        gO = cute.local_tile(mO[batch, None, head, None], (_BR, self._DO), (m_block, grp))
        cO = cute.local_tile(
            cute.make_identity_tensor((mO.shape[1], self._D)), (_BR, self._DO), (m_block, grp)
        )
        thr_o = copy_o.get_slice(gtid)
        tOsO = thr_o.partition_S(sO)
        tOgO = thr_o.partition_D(gO)
        tOcOg = thr_o.partition_D(cO)
        tOrO = cute.make_fragment_like(tOgO, cutlass.Float16)
        cute.copy(copy_o, tOsO, tOrO)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if cute.elem_less(tOcOg[0, m, 0][0], seqlen_q):
                cute.copy(copy_o, tOrO[None, m, None], tOgO[None, m, None])

    @cute.jit
    def _load_v(self, copy64, tVgV, tVcV, tVsV, num_tiles, i, s, grp, seqlen_k,
                first: cutlass.Constexpr):
        nb = num_tiles - 1 - i
        idx = s * 2 + grp
        if cutlass.const_expr(first):
            for m in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if cute.elem_less(tVcV[0, m, 0, nb][0], seqlen_k):
                    cute.copy(copy64, tVgV[None, m, None, nb], tVsV[None, m, None, idx])
                else:
                    tVsV[None, m, None, idx].fill(0)
        else:
            cute.copy(copy64, tVgV[None, None, None, nb], tVsV[None, None, None, idx])

    @cute.jit
    def _o_step(self, st, i, s, grp, num_tiles, seqlen_k, first: cutlass.Constexpr):
        (tOsVt, tOrVt, tOrVt_view, smem_copy_v, tOsP, tOrP, tOrP_view, smem_copy_p,
         tiled_mma, acc_O, tOcO_mn, sAlpha, copy64, tVgV, tVcV, tVsV) = st
        idx = s * 2 + grp

        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier(barrier_id=_BAR_O_GRP + grp, number_of_threads=64)
        cute.arch.barrier(barrier_id=_BAR_P_FULL + s, number_of_threads=_THREADS)

        if cutlass.const_expr(not first):
            acc_mn = self._mn_view(acc_O)
            for r in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                a = sAlpha[tOcO_mn[r, 0][0], s]
                acc_mn[r, None] = acc_mn[r, None].load() * a

        num_k = cute.size(tOsP.shape[2])
        cute.copy(smem_copy_p, tOsP[None, None, None, s], tOrP_view)
        cute.copy(smem_copy_v, tOsVt[None, None, 0, idx], tOrVt_view[None, None, 0])
        for k in cutlass.range_constexpr(num_k):
            if cutlass.const_expr(k + 1 < num_k):
                cute.copy(smem_copy_v, tOsVt[None, None, k + 1, idx], tOrVt_view[None, None, k + 1])
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)

        if i + 2 < num_tiles:
            cute.arch.barrier_arrive(barrier_id=_BAR_P_EMPTY + s, number_of_threads=_THREADS)
        cute.arch.barrier(barrier_id=_BAR_O_GRP + grp, number_of_threads=64)
        if i + 2 < num_tiles:
            self._load_v(copy64, tVgV, tVcV, tVsV, num_tiles, i + 2, s, grp, seqlen_k, False)
        cute.arch.cp_async_commit_group()

    # ---------------------------------------------------------------- helpers
    def _mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret a ``(MMA, MMA_M, MMA_N)`` accumulator as ``(M, N)``."""
        col_major = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            (
                (col_major.shape[0][1], col_major.shape[1]),
                (col_major.shape[0][0], col_major.shape[2]),
            ),
            stride=(
                (col_major.stride[0][1], col_major.stride[1]),
                (col_major.stride[0][0], col_major.stride[2]),
            ),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def _quad_max(self, val):
        val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        return cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))

    def _quad_sum(self, val):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        return val + cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
