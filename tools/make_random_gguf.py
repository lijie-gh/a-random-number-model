#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_random_gguf.py —— 零训练手搓「随机数」GGUF 模型（随机输出家族）
======================================================================

不管用户输入什么、上下文多长，模型只吐**一个**在指定范围里均匀随机的数，然后自己吐 EOS 停下。

原理（1 层 llama 架构，权重全部写死，不训练）：
  - attention / FFN 权重全 0  =>  残差流 = 上一个 token 的 embedding
  - embedding 只有两个取值：
        e_in  = one-hot(0)  一切"输入侧"token（BOS / UNK / 非数字的字节 token）
        e_num = one-hot(1)  每个"数字"token 自己
    所以隐藏状态只取决于"上一个 token 是不是数字"，与输入内容、上下文长度毫无关系
  - token_embd 与 output 解耦（tie_word_embeddings=false）
        out[:, 0] = -spread   只有数字 token 行 = +spread
        out[:, 1] = -spread   只有 EOS 行 = +spread
    两个状态：
        状态 A（上一个是 e_in）：logits 只在数字上取相同正值
            => softmax 是 N 个各 1/N 的均匀分布 => 采样器掷骰子 => 随机数
        状态 B（刚吐出数字，上一个是 e_num）：只有 EOS 是正值
            => 立刻吐 EOS => 生成自动停，只输出一个数

  "随机"不是学出来的：它是 softmax 均匀分布 + 采样器的随机性。
  "自动停"是真的 EOS：不需要 num_predict=1 也只吐一个数。

词表约束（硬要求，违反会 GGML_ASSERT 崩）：
  同一个 token 串不能出现两次（id_to_token.size() == token_to_id.size()）。
  所以 0~9 这种单字符数字必须复用已有的 256 个 gpt2 字节 token；
  10~100 这种多字符数字才新建内容 token（空 merges 下编码器永远吐不出它们）。

用法：
  python make_random_gguf.py <输出.gguf> --family 0100 [--dim 256] [--spread 100]
  python make_random_gguf.py --inspect <任意.gguf>

family: 01 = 0/1；09 = 0~9；0100 = 0~100；d6 = 1~6（骰子）；也可 --numbers "1,7,42"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import make_fixed_gguf as G

SPECIALS = G.SPECIALS                       # 0=<unk> 1=<s> 2=</s>（EOS 就是"停"信号）
BYTES = G.BYTE_TOKEN_STRS                   # 256 个 gpt2 字节 token
N_SPECIAL = len(SPECIALS)
ID_UNK, ID_BOS, ID_EOS = 0, 1, 2

FAMILIES = {
    "01": ["0", "1"],
    "09": [str(i) for i in range(10)],
    "0100": [str(i) for i in range(101)],
    "d6": [str(i) for i in range(1, 7)],
    "d20": [str(i) for i in range(1, 21)],
}


def build_weights(numbers, dim, spread):
    """返回 (tokens, types, emb, out, norm, num_ids, safe_ids)。"""
    assert len(set(numbers)) == len(numbers), "取值有重复"

    # 单字符 -> 复用字节 token；多字符 -> 新建内容 token（空 merges 下编码器产不出来）
    single = [n for n in numbers if len(n) == 1]
    multi = [n for n in numbers if len(n) > 1]
    for n in single:
        assert len(n.encode("utf-8")) == 1, f"单字符取值 {n!r} 不是单个字节"
    body = [G.enc_tok(n.encode("utf-8")) for n in multi]
    assert len(set(body)) == len(body), "内容 token 有重复"
    assert not (set(body) & set(BYTES)), "内容 token 与字节 token 撞名"

    tokens = SPECIALS + body + BYTES
    V = len(tokens)
    types = [2, 3, 3] + [1] * (V - 3)

    num_ids = []
    for n in single:
        num_ids.append(N_SPECIAL + len(body) + BYTES.index(G.enc_tok(n.encode("utf-8"))))
    for i in range(len(body)):
        num_ids.append(N_SPECIAL + i)
    # 数字按输入顺序记录：single 在前、multi 在后，统一按 numbers 顺序重排
    by_str = {}
    for n in single:
        by_str[n] = N_SPECIAL + len(body) + BYTES.index(G.enc_tok(n.encode("utf-8")))
    for i, n in enumerate(multi):
        by_str[n] = N_SPECIAL + i
    num_ids = [by_str[n] for n in numbers]
    assert len(set(num_ids)) == len(num_ids)
    # 硬校验：数字 id 上必须正好是对应的 token 串
    assert [tokens[i] for i in num_ids] == [G.enc_tok(n.encode("utf-8")) for n in numbers], \
        "数字 token 位置错位"

    safe_ids = [i for i in range(V) if i not in set(num_ids)]

    e_in = np.zeros(dim, np.float32)
    e_in[0] = 1.0
    e_num = np.zeros(dim, np.float32)
    e_num[1] = 1.0

    emb = np.tile(e_in, (V, 1))            # 默认：一切非数字 token 都指向"状态 A"
    emb[num_ids] = e_num                   # 数字 token 指向"状态 B"（下一步出 EOS）
    out = np.zeros((V, dim), np.float32)
    out[:, 0] = -spread
    out[:, 1] = -spread
    out[num_ids, 0] = +spread              # 状态 A：只有数字被抬起
    out[ID_EOS, 1] = +spread               # 状态 B：只有 EOS 被抬起
    norm = np.ones(dim, np.float32)
    return tokens, types, emb, out, norm, num_ids, safe_ids


def forward(emb, out, norm, tok, eps=1e-5):
    x = emb[tok].astype(np.float64)
    h = x / np.sqrt(np.mean(x * x) + eps) * norm
    return out.astype(np.float64) @ h


def selftest(numbers, emb, out, norm, num_ids, safe_ids):
    ok = True
    N = len(num_ids)
    lg_a = forward(emb, out, norm, ID_BOS)
    top = np.argsort(-lg_a)
    if sorted(int(i) for i in top[:N]) != sorted(num_ids):
        print(f"[自测] FAIL 状态 A 的 top-{N} 不是全部数字 token")
        ok = False
    vals = {round(float(lg_a[i]), 6) for i in num_ids}
    if len(vals) != 1:
        print(f"[自测] FAIL 数字之间 logit 不相等：{sorted(vals)[:5]}")
        ok = False
    gap = float(lg_a[top[N - 1]] - lg_a[top[N]])
    p = np.exp(lg_a - lg_a.max())
    p /= p.sum()
    pn = p[num_ids]
    print(f"[自测] 状态A 数字 logit {lg_a[num_ids[0]]:+.2f} / 非数字最高 {lg_a[top[N]]:+.2f}"
          f"  间隔 {gap:.1f}  数字概率 {pn.min():.6f}~{pn.max():.6f}（理想 1/{N}={1/N:.6f}）"
          f"  其余合计 {1-pn.sum():.2e}")

    for tid in (num_ids[0], num_ids[N // 2], num_ids[-1]):
        lg_b = forward(emb, out, norm, tid)
        if int(np.argmax(lg_b)) != ID_EOS:
            print(f"[自测] FAIL 状态 B（上一个 token={tid}）argmax={int(np.argmax(lg_b))} 不是 EOS")
            ok = False
    lg_b = forward(emb, out, norm, num_ids[0])
    order_b = np.argsort(-lg_b)
    print(f"[自测] 状态B EOS logit {lg_b[ID_EOS]:+.2f} / 次高 {lg_b[order_b[1]]:+.2f}"
          f"  间隔 {float(lg_b[order_b[0]] - lg_b[order_b[1]]):.1f}")

    bad = [t for t in safe_ids if not np.allclose(forward(emb, out, norm, t), lg_a)]
    if bad:
        print(f"[自测] FAIL {len(bad)} 个非数字 token 会改变状态，例如 {bad[:5]}")
        ok = False
    print(f"[自测] {len(safe_ids)} 个非数字 token 全部落到同一状态：{'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="零训练手搓「随机数」GGUF")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--family", default="0100", choices=sorted(FAMILIES))
    ap.add_argument("--numbers", default=None, help='自定义取值，逗号分隔，覆盖 --family')
    ap.add_argument("--layers", type=int, default=1, choices=(0, 1))
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--spread", type=float, default=100.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--inspect", metavar="GGUF")
    args = ap.parse_args()

    if args.inspect:
        G.inspect(args.inspect)
        return
    if not args.output:
        ap.error("需要输出 .gguf（或用 --inspect）")

    numbers = ([s.strip() for s in args.numbers.split(",") if s.strip()]
               if args.numbers else FAMILIES[args.family])
    d, n_ff, L = args.dim, args.ff, args.layers
    tokens, types, emb, out, norm, num_ids, safe_ids = build_weights(numbers, d, args.spread)
    V = len(tokens)

    tensors = [
        ("token_embd.weight", emb, G.GGML_TYPE_F16),
        ("output.weight", out, G.GGML_TYPE_F16),
        ("output_norm.weight", norm, G.GGML_TYPE_F32),
    ]
    if L >= 1:
        zero_dd = np.zeros((d, d), np.float32)
        tensors += [
            ("blk.0.attn_norm.weight", np.ones(d, np.float32), G.GGML_TYPE_F32),
            ("blk.0.ffn_norm.weight", np.ones(d, np.float32), G.GGML_TYPE_F32),
            ("blk.0.attn_q.weight", zero_dd, G.GGML_TYPE_F16),
            ("blk.0.attn_k.weight", zero_dd, G.GGML_TYPE_F16),
            ("blk.0.attn_v.weight", zero_dd, G.GGML_TYPE_F16),
            ("blk.0.attn_output.weight", zero_dd, G.GGML_TYPE_F16),
            ("blk.0.ffn_gate.weight", np.zeros((n_ff, d), np.float32), G.GGML_TYPE_F16),
            ("blk.0.ffn_up.weight", np.zeros((n_ff, d), np.float32), G.GGML_TYPE_F16),
            ("blk.0.ffn_down.weight", np.zeros((d, n_ff), np.float32), G.GGML_TYPE_F16),
        ]

    name = args.name or f"random-{numbers[0]}-{numbers[-1]}"
    kvs = [
        ("general.architecture", G.T_STR, "llama"),
        ("general.name", G.T_STR, name),
        ("general.description", G.T_STR,
         f"Handcrafted zero-training toy model: emits exactly one uniformly random value out of "
         f"{len(numbers)} options ({numbers[0]}..{numbers[-1]}) and then stops with EOS, regardless "
         f"of the input or the context. All non-number embeddings are identical; the output matrix "
         f"raises only the option tokens and the EOS."),
        ("general.file_type", G.T_U32, 1),
        ("general.alignment", G.T_U32, G.ALIGNMENT),
        ("llama.context_length", G.T_U32, 8192),
        ("llama.embedding_length", G.T_U32, d),
        ("llama.block_count", G.T_U32, L),
        ("llama.feed_forward_length", G.T_U32, n_ff),
        ("llama.attention.head_count", G.T_U32, 1),
        ("llama.attention.head_count_kv", G.T_U32, 1),
        ("llama.attention.key_length", G.T_U32, d),
        ("llama.attention.value_length", G.T_U32, d),
        ("llama.attention.layer_norm_rms_epsilon", G.T_F32, 1e-5),
        ("llama.rope.dimension_count", G.T_U32, d),
        ("llama.rope.freq_base", G.T_F32, 10000.0),
        ("llama.vocab_size", G.T_U32, V),
        ("llama.tie_word_embeddings", G.T_BOOL, False),
        ("tokenizer.ggml.model", G.T_STR, "gpt2"),
        # 不写这个键，新版 llama.cpp 会在加载时警告 missing pre-tokenizer（"GENERATION QUALITY
        # WILL BE DEGRADED!"）。本模型只按字节切词，加了也完全等价，但清掉告警更像成品。
        ("tokenizer.ggml.pre", G.T_STR, "default"),
        ("tokenizer.ggml.tokens", G.T_ARR, (G.T_STR, tokens)),
        ("tokenizer.ggml.token_type", G.T_ARR, (G.T_I32, types)),
        ("tokenizer.ggml.merges", G.T_ARR, (G.T_STR, [])),
        ("tokenizer.ggml.bos_token_id", G.T_U32, ID_BOS),
        ("tokenizer.ggml.eos_token_id", G.T_U32, ID_EOS),
        ("tokenizer.ggml.unknown_token_id", G.T_U32, ID_UNK),
        ("tokenizer.ggml.add_bos_token", G.T_BOOL, True),
        ("tokenizer.ggml.add_eos_token", G.T_BOOL, False),
        # 模板只吐 BOS：正文在模板层就被丢掉（双保险，也避开"输入以数字结尾"的边角）
        ("tokenizer.chat_template", G.T_STR, "{{ bos_token }}"),
    ]
    G.write_gguf(args.output, kvs, tensors)

    n_emb_out = 2 * V * d
    n_blk = (4 * d * d + 3 * n_ff * d) if L >= 1 else 0
    size = Path(args.output).stat().st_size
    print(f"[生成] 取值 {numbers[0]}..{numbers[-1]} 共 {len(numbers)} 个；层 {L}；dim {d}；ff {n_ff}；"
          f"词表 {V}（内容 {len(tokens) - 3 - 256} + 字节 256 + 特殊 3）")
    print(f"[生成] 参数 {n_emb_out + n_blk + d:,}；文件 {size:,} 字节 -> {args.output}")
    ok = selftest(numbers, emb, out, norm, num_ids, safe_ids)
    print(f"[结论] {'通过' if ok else '失败'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

