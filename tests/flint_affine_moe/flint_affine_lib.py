"""Helpers for the Flint affine-INT4 Marlin MoE kernel tests: checkpoint loader (MLX affine INT4, group 64), the
Python reference repack into the Marlin layout the cloned kernel expects, and the FP32 reference MoE.

Weight format (per layer, stacked experts):
  *.weight  uint8  [E, N, K/2]   low nibble = even k, high nibble = odd k, K-contiguous per output row
  *.scales  bf16   [E, N, K/64]
  *.biases  bf16   [E, N, K/64]
  w[e,n,k] = scales[e,n,k//64] * q[e,n,k] + biases[e,n,k//64]
Kernel formulation: w = (q - 8) * s16 + b16,  b16 = fp16(bias + 8 * scale) (fp32 sum, one rounding).
"""
import glob, json, mmap, struct
import numpy as np
import torch

DEV = "cuda"
GROUP = 64


class Ckpt:
    def __init__(self, d):
        self.idx, self.maps = {}, {}
        for f in sorted(glob.glob(f"{d}/model-*.safetensors")):
            fh = open(f, "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps[f] = (fh, mm)
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.idx[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                                   v["dtype"], v["shape"])

    def get(self, name, dev=DEV):
        f, off, size, dt, shape = self.idx[name]
        np_dt = {"BF16": np.uint16, "U8": np.uint8, "F16": np.float16, "F32": np.float32}[dt]
        arr = np.frombuffer(self.maps[f][1], dtype=np_dt, count=size // np.dtype(np_dt).itemsize, offset=off)
        t = torch.from_numpy(np.ascontiguousarray(arr.reshape(shape))).to(dev)
        return t.view(torch.bfloat16) if dt == "BF16" else t


def unpack_q(w_u8):
    """uint8 [E,N,K/2] -> int32 q [E,N,K] (low nibble = even k)."""
    return torch.stack([w_u8 & 0xF, w_u8 >> 4], dim=-1).flatten(-2).to(torch.int32)


def dequant_fp32(w_u8, s_bf16, b_bf16):
    q = unpack_q(w_u8).to(torch.float32).unflatten(-1, (-1, GROUP))
    s = s_bf16.to(torch.float32).unsqueeze(-1)
    b = b_bf16.to(torch.float32).unsqueeze(-1)
    return (q * s + b).flatten(-2)


def make_sb(s_bf16, b_bf16):
    """fp16 scale and b' = bias + 8*scale (fp32 sum, one rounding to fp16). Shapes [E,N,G]."""
    s32 = s_bf16.to(torch.float32)
    bp32 = b_bf16.to(torch.float32) + 8.0 * s32
    return s32.to(torch.float16), bp32.to(torch.float16)


# ---- Marlin layout index functions (same as cpp/kernels/weightsTransform/int4/moe/gptqMarlin/marlinRepackIndex.cuh)
def _pack_index(i): return 2 * (i % 4) + i // 4
def _output_index(slot): return 4 * (slot % 32) + slot // 32
def _row_index(slot, pc): return 2 * (slot % 4) + (pc % 2) + 8 * ((pc // 2) % 2)
def _col_index(slot, pc):
    block = slot // 4
    return (block // 8) * 16 + block % 8 + (0 if pc < 4 else 8)


def _tile_tables():
    rr = np.zeros((128, 8), np.int64); cc = np.zeros((128, 8), np.int64)
    for slot in range(128):
        for p in range(8):
            ps = _pack_index(p)
            rr[slot, p] = _row_index(slot, ps); cc[slot, p] = _col_index(slot, ps)
    out_pos = np.array([_output_index(s) for s in range(128)])
    return rr, cc, out_pos


def marlin_repack_q(q_ekn, e_chunk=8):
    """q int32 [E, N, K] (values 0..15) -> int32 [E, K/16, N*2] Marlin tile layout (kU4, w = q - 8 in kernel).
    Output word (kt, nt*128 + outputIndex(slot)) holds, in nibble p, q[n = nt*64 + col(slot,p), k = kt*16 + row(slot,p)]."""
    E, N, K = q_ekn.shape
    rr, cc, out_pos = _tile_tables()
    rr_t = torch.from_numpy(rr).to(q_ekn.device); cc_t = torch.from_numpy(cc).to(q_ekn.device)
    flat = (rr_t * 64 + cc_t)                                    # [128, 8] index into the 16x64 tile
    shifts = (4 * torch.arange(8, device=q_ekn.device, dtype=torch.int32))
    out = torch.empty((E, K // 16, N * 2), dtype=torch.int32, device=q_ekn.device)
    perm = torch.from_numpy(out_pos).to(q_ekn.device)
    for e0 in range(0, E, e_chunk):
        q = q_ekn[e0:e0 + e_chunk].transpose(1, 2)               # [e, K, N]
        e = q.shape[0]
        t = q.reshape(e, K // 16, 16, N // 64, 64).permute(0, 1, 3, 2, 4).reshape(e, K // 16, N // 64, 1024)
        g = t[..., flat.reshape(-1)].reshape(e, K // 16, N // 64, 128, 8)
        words = (g << shifts).sum(-1, dtype=torch.int32)         # nibbles never overlap; int32 wrap is fine
        res = torch.empty_like(words)
        res[..., perm] = words                                   # word of `slot` goes to column outputIndex(slot)
        out[e0:e0 + e] = res.reshape(e, K // 16, N * 2)
    return out


def marlin_permute_scales(s_egn):
    """[E, G, N] -> same shape with in-block-of-64 n permuted (matches int4MoeScaleRepack.cu cScalePermutation)."""
    E, G, N = s_egn.shape
    n = torch.arange(N, device=s_egn.device)
    out_n = (n // 64) * 64 + 8 * (n % 8) + (n % 64) // 8
    out = torch.empty_like(s_egn)
    out[:, :, out_n] = s_egn
    return out


def repack_proj_pair(ckpt, prefix_a, prefix_b=None, e_chunk=8):
    """Repack one projection (or the fused gate+up pair: columns [gate | up]) for the cloned kernel.
    Returns qw int32 [E,K/16,2N], s fp16 [E,G,N], bp fp16 [E,G,N] (scales/b' Marlin-permuted)."""
    parts = [prefix_a] + ([prefix_b] if prefix_b else [])
    ws = [ckpt.get(p + ".weight") for p in parts]
    ss, bps = [], []
    for p in parts:
        s16, bp16 = make_sb(ckpt.get(p + ".scales"), ckpt.get(p + ".biases"))
        ss.append(s16.transpose(1, 2)); bps.append(bp16.transpose(1, 2))     # [E,G,N]
    s = torch.cat(ss, dim=2); bp = torch.cat(bps, dim=2)
    E = ws[0].shape[0]
    outs = []
    for e0 in range(0, E, e_chunk):                                          # unpack per expert chunk (int32 is big)
        q = torch.cat([unpack_q(w[e0:e0 + e_chunk]) for w in ws], dim=1)     # [e, N_total, K]
        outs.append(marlin_repack_q(q, e_chunk))
    return torch.cat(outs, 0), marlin_permute_scales(s).contiguous(), marlin_permute_scales(bp).contiguous()


def routing(logits_fp32, topk=6):
    """Sigmoid top-k, renormalised over the selected weights (no groups, no bias)."""
    sc = torch.sigmoid(logits_fp32)
    w, ids = sc.topk(topk, dim=-1)
    return w / (w.sum(-1, keepdim=True) + 1e-20), ids


def build_marlin_indices(ids, num_experts, block):
    """ids [T, topk] int -> sorted_token_ids (padded slots = T*topk sentinel), expert_ids per block, num_post_padded."""
    T, K = ids.shape
    total = T * K
    flat = ids.flatten().cpu().numpy()
    sorted_ids, exp_ids = [], []
    for e in range(num_experts):
        slots = np.nonzero(flat == e)[0]
        if len(slots) == 0:
            continue
        pad = (-len(slots)) % block
        sorted_ids += list(slots) + [total] * pad
        exp_ids += [e] * ((len(slots) + pad) // block)
    return (torch.tensor(sorted_ids, dtype=torch.int32), torch.tensor(exp_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int32))


def ref_moe(h, w_gate, w_up, w_down, wts, ids):
    """FP32 reference. w_*: dequantised fp32 [E,N,K]; h fp32 [T,H]. Returns [T,H]."""
    y = torch.zeros_like(h)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        he = h[tok]
        a = torch.nn.functional.silu(he @ w_gate[e].T) * (he @ w_up[e].T)
        y.index_add_(0, tok, (a @ w_down[e].T) * wts[tok, slot][:, None])
    return y
