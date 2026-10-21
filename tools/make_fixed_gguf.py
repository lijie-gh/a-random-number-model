#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fixed_gguf.py —— 零训练手搓「固定输出」GGUF 模型
======================================================

无论输入什么，模型都逐字节输出指定文件的全部内容。
原理（1 层 llama 架构，权重全部写死，不训练）：

  - attention / FFN 权重全部置 0  =>  残差流 = 上一个 token 的 embedding
  - token_embd 与 output（不共享）组成单向链表：
        emb[t] 对齐 output 矩阵的 next(t) 行
    => 每步 argmax 唯一确定地指向下一个 token
  - BOS / 字节兜底 token 的 embedding 一律指向第一个内容 token
  - 最后一个内容 token 指向 EOS => 生成完自动停

输出是标准 GGUF v3，llama.cpp / Ollama / LM Studio 可直接加载。

用法：
  python make_fixed_gguf.py <输入文件> <输出.gguf> [--dim 64] [--ff 32] [--target 40]
  python make_fixed_gguf.py --inspect <任意.gguf>     （解剖结构，自检用）

要支持量化：Q8_0/Q5_0/Q4_0 族要求每个张量行宽是 32 的倍数（默认 dim=64、ff=32 即满足）；
K 系量化（Q4_K_M 等）要求 256 的倍数，需 --dim 256 --ff 256。
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

T_U8, T_I8, T_U16, T_I16 = 0, 1, 2, 3
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR = 4, 5, 6, 7, 8, 9
T_U64, T_I64, T_F64 = 10, 11, 12
SCALAR_SIZE = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
               T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}

ALIGNMENT = 32


# ----------------------------------------------------------- GPT-2 字节编码表
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}
BYTE_TOKEN_STRS = [B2U[b] for b in range(256)]
SPECIALS = ["<unk>", "<s>", "</s>"]


def enc_tok(raw: bytes) -> str:
    """把一段字节变成 GGUF 词表里 gpt2 风格的 token 字符串。"""
    return "".join(B2U[b] for b in raw)


# ----------------------------------------------------------- 文本切块（去重）
def make_chunks(text, target=40, hard_max=96):
    segs = []
    for ln in text.splitlines(keepends=True):
        if len(ln) <= hard_max:
            segs.append(ln)
        else:
            segs += [ln[i:i + hard_max] for i in range(0, len(ln), hard_max)]
    chunks, buf = [], ""
    for s in segs:
        buf += s
        if len(buf) >= target:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def ensure_unique(chunks, reserved):
    """把所有块改成两两唯一（重复块向后合并），保证词表无重复。"""
    chunks = list(chunks)
    while True:
        seen = set(reserved)
        out, changed, i = [], False, 0
        while i < len(chunks):
            c = chunks[i]
            if c in seen and i + 1 < len(chunks):
                chunks[i + 1] = c + chunks[i + 1]
                changed = True
            elif c in seen:
                if not out:
                    raise RuntimeError("无法去重：整个内容只有一个块且与保留词冲突")
                out[-1] += c
                changed = True
            else:
                seen.add(c)
                out.append(c)
            i += 1
        if not changed:
            return out
        chunks = out


# ----------------------------------------------------------- 构造链表权重
def build_tensors(C, V, dim, seed):
    rng = np.random.default_rng(seed)

    def rand_unit():
        x = rng.standard_normal(dim)
        return (x / np.linalg.norm(x)).astype(np.float32)

    content_ids = list(range(3, 3 + C))          # 0=unk 1=bos 2=eos
    vec = {cid: rand_unit() for cid in content_ids}
    vec[2] = rand_unit()                          # EOS 的"指向"向量
    v_start = vec[content_ids[0]]

    emb = np.zeros((V, dim), dtype=np.float32)
    out = np.zeros((V, dim), dtype=np.float32)

    # 链表核心：emb[t] 对齐 out[next(t)] 行（目标行放大 8 倍、噪声行缩小到 0.5 倍，把 argmax 间隔拉到最大）
    for i, cid in enumerate(content_ids):
        nxt = content_ids[i + 1] if i + 1 < C else 2
        emb[cid] = vec[nxt]
        out[cid] = 8.0 * vec[cid]
    out[2] = 8.0 * vec[2]

    # 所有非内容 token（BOS/UNK/256 个字节 token）：一律指向第一个内容 token
    for tid in range(V):
        if tid in vec:
            continue
        emb[tid] = v_start
        out[tid] = 0.5 * rand_unit()
    emb[2] = v_start
    return emb, out


# ----------------------------------------------------------- GGUF 写出
def enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_scalar(vtype, v):
    if vtype == T_U32:
        return struct.pack("<I", int(v))
    if vtype == T_I32:
        return struct.pack("<i", int(v))
    if vtype == T_F32:
        return struct.pack("<f", float(v))
    if vtype == T_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if vtype == T_STR:
        return enc_string(v)
    raise ValueError(f"不支持的标量类型 {vtype}")


def enc_kv(key, vtype, value) -> bytes:
    blob = enc_string(key) + struct.pack("<I", vtype)
    if vtype == T_ARR:
        etype, vals = value
        blob += struct.pack("<I", etype) + struct.pack("<Q", len(vals))
        if etype == T_STR:
            for s in vals:
                blob += enc_string(s)
        else:
            for x in vals:
                blob += enc_scalar(etype, x)
    else:
        blob += enc_scalar(vtype, value)
    return blob


def write_gguf(path, kvs, tensors):
    """tensors: list of (name, np.ndarray, ggml_type)；dims 按 ggml 约定反序写入。"""
    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", GGUF_VERSION)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", len(kvs))
    for k, t, v in kvs:
        buf += enc_kv(k, t, v)

    infos = bytearray()
    data_items = []
    cur = 0
    for name, arr, gtype in tensors:
        arr = np.ascontiguousarray(arr)
        raw = arr.astype(np.float16).tobytes() if gtype == GGML_TYPE_F16 else arr.astype(np.float32).tobytes()
        cur = (cur + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        infos += enc_string(name)
        dims = tuple(reversed(arr.shape))
        infos += struct.pack("<I", len(dims))
        for dd in dims:
            infos += struct.pack("<Q", dd)
        infos += struct.pack("<I", gtype)
        infos += struct.pack("<Q", cur)
        data_items.append((raw, cur))
        cur += len(raw)
    buf += infos
    pad = (-len(buf)) % ALIGNMENT
    buf += b"\x00" * pad
    data_start = len(buf)
    for raw, off in data_items:
        gap = (data_start + off) - len(buf)
        assert gap >= 0
        buf += b"\x00" * gap
        buf += raw
    Path(path).write_bytes(bytes(buf))


# ----------------------------------------------------------- GGUF 读取（自检用）
def read_gguf(path):
    data = Path(path).read_bytes()
    pos = 0

    def take(n):
        nonlocal pos
        b = data[pos:pos + n]
        pos += n
        return b

    if struct.unpack("<I", take(4))[0] != GGUF_MAGIC:
        raise ValueError("不是 GGUF 文件")
    ver = struct.unpack("<I", take(4))[0]
    n_tensors = struct.unpack("<Q", take(8))[0]
    n_kv = struct.unpack("<Q", take(8))[0]

    def read_val(t):
        if t == T_U32:
            return struct.unpack("<I", take(4))[0]
        if t == T_I32:
            return struct.unpack("<i", take(4))[0]
        if t == T_F32:
            return struct.unpack("<f", take(4))[0]
        if t == T_BOOL:
            return take(1) != b"\x00"
        if t == T_STR:
            n = struct.unpack("<Q", take(8))[0]
            return take(n).decode("utf-8", errors="replace")
        if t == T_ARR:
            et = struct.unpack("<I", take(4))[0]
            cnt = struct.unpack("<Q", take(8))[0]
            return [read_val(et) for _ in range(cnt)]
        if t in SCALAR_SIZE:
            return take(SCALAR_SIZE[t])
        raise ValueError(f"未知类型 {t}")

    kvs = {}
    for _ in range(n_kv):
        n = struct.unpack("<Q", take(8))[0]
        key = take(n).decode("utf-8")
        vt = struct.unpack("<I", take(4))[0]
        kvs[key] = read_val(vt)

    infos = {}
    for _ in range(n_tensors):
        n = struct.unpack("<Q", take(8))[0]
        name = take(n).decode("utf-8")
        nd = struct.unpack("<I", take(4))[0]
        dims = [struct.unpack("<Q", take(8))[0] for _ in range(nd)]
        gt = struct.unpack("<I", take(4))[0]
        off = struct.unpack("<Q", take(8))[0]
        infos[name] = (dims, gt, off)

    align = kvs.get("general.alignment", ALIGNMENT)
    data_start = (pos + align - 1) // align * align

    def load(name):
        dims, gt, off = infos[name]
        shape = tuple(reversed(dims))
        cnt = 1
        for s in shape:
            cnt *= s
        if gt == GGML_TYPE_F32:
            a = np.frombuffer(data, dtype="<f4", count=cnt, offset=data_start + off)
        elif gt == GGML_TYPE_F16:
            a = np.frombuffer(data, dtype="<f2", count=cnt, offset=data_start + off)
        else:
            raise ValueError(f"暂不支持读取 ggml 类型 {gt}")
        return a.reshape(shape).astype(np.float32)

    return kvs, infos, load, ver


# ----------------------------------------------------------- 自测
def selftest(path, original: bytes):
    kvs, infos, load, ver = read_gguf(path)
    tokens = kvs["tokenizer.ggml.tokens"]
    eos = kvs["tokenizer.ggml.eos_token_id"]
    bos = kvs["tokenizer.ggml.bos_token_id"]
    emb = load("token_embd.weight")
    out = load("output.weight")
    norm = load("output_norm.weight")
    eps = kvs["llama.attention.layer_norm_rms_epsilon"]
    C = len(tokens) - 3 - 256

    def next_id(prev):
        x = emb[prev]
        u = x / np.sqrt(np.mean(x * x) + eps) * norm
        logits = out @ u
        order = np.argsort(logits)
        return int(order[-1]), float(logits[order[-1]] - logits[order[-2]])

    def run(start):
        ids, gaps, t = [], [], start
        for _ in range(C + 16):
            t, g = next_id(t)
            gaps.append(g)
            if t == eos:
                return ids, gaps[:-1]
            ids.append(t)
        raise RuntimeError("没有在合理步数内遇到 EOS")

    def detok(ids):
        raw = bytearray()
        for i in ids:
            for ch in tokens[i]:
                raw.append(U2B[ch])
        return bytes(raw)

    starts = [("BOS", bos), ("UNK", 0),
              ("byte 'h'", 3 + C + ord("h")),
              ("byte '<'", 3 + C + ord("<")),
              ("byte 0xFF", 3 + C + 255)]
    all_ok = True
    for name, s in starts:
        ids, gaps = run(s)
        got = detok(ids)
        ok = got == original
        all_ok &= ok
        print(f"[自测] 起点 {name:<9} 输出 {len(ids):>4} token  逐字节一致: {'OK ' if ok else 'FAIL'}"
              f"  最小 logit 间隔 {min(gaps):.2f}")
        if not ok:
            i = next((k for k in range(min(len(got), len(original))) if got[k] != original[k]),
                     min(len(got), len(original)))
            print(f"       第一处差异 @ {i}:")
            print(f"       得到 {got[max(0, i - 24):i + 24]!r}")
            print(f"       期望 {original[max(0, i - 24):i + 24]!r}")
    return all_ok


# ----------------------------------------------------------- inspect 模式
def inspect(path):
    kvs, infos, _, ver = read_gguf(path)
    print(f"== GGUF v{ver}  {path}")
    print(f"文件大小 {Path(path).stat().st_size:,} 字节, 张量 {len(infos)} 个, 元数据 {len(kvs)} 条")
    for k in sorted(kvs):
        v = kvs[k]
        if isinstance(v, list):
            show = f"[{len(v)} 项] " + repr(v[:3])[:60]
        else:
            show = repr(v)[:90]
        print(f"  {k} = {show}")
    print("-- 张量 --")
    for name in sorted(infos):
        dims, gt, off = infos[name]
        gt_name = {0: "F32", 1: "F16"}.get(gt, f"type{gt}")
        print(f"  {name:<28} {gt_name:<5} dims={dims} offset={off}")


# ----------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="零训练手搓固定输出 GGUF")
    ap.add_argument("input", nargs="?", help="要固定输出的文本文件")
    ap.add_argument("output", nargs="?", help="输出 .gguf")
    ap.add_argument("--dim", type=int, default=64, help="嵌入维度，默认 64")
    ap.add_argument("--ff", type=int, default=32, help="FFN 中间维，默认 32（须为 32 的倍数才能被 Q8_0/Q4_0 族量化；K 系量化需 256）")
    ap.add_argument("--target", type=int, default=40, help="切块目标长度，默认 40 字符")
    ap.add_argument("--hard-max", type=int, default=96, help="单块最大长度，默认 96 字符")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--inspect", metavar="GGUF", help="解剖一个 GGUF 文件后退出")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    if not args.input or not args.output:
        ap.error("需要 输入文件 和 输出.gguf（或用 --inspect）")

    raw = Path(args.input).read_bytes()
    text = raw.decode("utf-8")

    chunks = ensure_unique(make_chunks(text, args.target, args.hard_max),
                           set(BYTE_TOKEN_STRS) | set(SPECIALS))
    assert "".join(chunks) == text, "切块后无法还原原文！"
    assert len(set(chunks)) == len(chunks), "切块有重复！"
    C = len(chunks)

    tokens = SPECIALS + [enc_tok(c.encode("utf-8")) for c in chunks] + BYTE_TOKEN_STRS
    types = [2, 3, 3] + [1] * C + [1] * 256
    V = len(tokens)

    emb, out = build_tensors(C, V, args.dim, args.seed)
    d, n_ff = args.dim, args.ff
    zero_dd = np.zeros((d, d), np.float32)

    tensors = [
        ("token_embd.weight", emb, GGML_TYPE_F16),
        ("output.weight", out, GGML_TYPE_F16),
        ("output_norm.weight", np.ones(d, np.float32), GGML_TYPE_F32),
        ("blk.0.attn_norm.weight", np.ones(d, np.float32), GGML_TYPE_F32),
        ("blk.0.ffn_norm.weight", np.ones(d, np.float32), GGML_TYPE_F32),
        ("blk.0.attn_q.weight", zero_dd, GGML_TYPE_F16),
        ("blk.0.attn_k.weight", zero_dd, GGML_TYPE_F16),
        ("blk.0.attn_v.weight", zero_dd, GGML_TYPE_F16),
        ("blk.0.attn_output.weight", zero_dd, GGML_TYPE_F16),
        ("blk.0.ffn_gate.weight", np.zeros((n_ff, d), np.float32), GGML_TYPE_F16),
        ("blk.0.ffn_down.weight", np.zeros((d, n_ff), np.float32), GGML_TYPE_F16),
        ("blk.0.ffn_up.weight", np.zeros((n_ff, d), np.float32), GGML_TYPE_F16),
    ]

    kvs = [
        ("general.architecture", T_STR, "llama"),
        ("general.name", T_STR, "pelican-fixed"),
        ("general.description", T_STR,
         "Handcrafted zero-training model: always outputs a fixed file, byte for byte. "
         "Weights are a hand-built linked list, not trained."),
        ("general.file_type", T_U32, 1),
        ("general.alignment", T_U32, ALIGNMENT),
        ("llama.context_length", T_U32, 8192),
        ("llama.embedding_length", T_U32, d),
        ("llama.block_count", T_U32, 1),
        ("llama.feed_forward_length", T_U32, n_ff),
        ("llama.attention.head_count", T_U32, 1),
        ("llama.attention.head_count_kv", T_U32, 1),
        ("llama.attention.key_length", T_U32, d),
        ("llama.attention.value_length", T_U32, d),
        ("llama.attention.layer_norm_rms_epsilon", T_F32, 1e-5),
        ("llama.rope.dimension_count", T_U32, d),
        ("llama.rope.freq_base", T_F32, 10000.0),
        ("llama.vocab_size", T_U32, V),
        ("llama.tie_word_embeddings", T_BOOL, False),
        ("tokenizer.ggml.model", T_STR, "gpt2"),
        ("tokenizer.ggml.tokens", T_ARR, (T_STR, tokens)),
        ("tokenizer.ggml.token_type", T_ARR, (T_I32, types)),
        ("tokenizer.ggml.merges", T_ARR, (T_STR, [])),
        ("tokenizer.ggml.bos_token_id", T_U32, 1),
        ("tokenizer.ggml.eos_token_id", T_U32, 2),
        ("tokenizer.ggml.unknown_token_id", T_U32, 0),
        ("tokenizer.ggml.add_bos_token", T_BOOL, True),
        ("tokenizer.ggml.add_eos_token", T_BOOL, False),
        ("tokenizer.chat_template", T_STR, "{{ bos_token }}"),
    ]

    write_gguf(args.output, kvs, tensors)

    params = V * d * 2 + (4 * d * d + 3 * n_ff * d) + 3 * d
    size = Path(args.output).stat().st_size
    print(f"[生成] {args.input}  ({len(raw):,} 字节)")
    print(f"[生成] 词表 {V} = 内容块 {C} + 字节兜底 256 + 特殊 3")
    print(f"[生成] 参数约 {params:,} 个，文件 {size:,} 字节 -> {args.output}")
    ok = selftest(args.output, raw)
    print(f"[结论] 全部自测 {'通过' if ok else '失败'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
