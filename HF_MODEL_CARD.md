---
license: mit
language:
  - en
tags: [gguf, llama.cpp, ollama, random-number, toy-model, zero-training, tiny]
pipeline_tag: text-generation
library_name: llama.cpp
---

# random-number-gguf

A GGUF model that does exactly one thing: **return a uniformly random number** in a configured range,
regardless of what you ask. It ignores the prompt, the context, and the system message, emits a single
number, and stops.

| Version | Range | Vocab | Params | F16 | Q2_K |
|---|---|---|---|---|---|
| `random-binary` | 0 or 1 | 259 | 591,616 | 1.16 MB | 249 KB |
| `random-digit` | 0–9 | 259 | 591,616 | 1.16 MB | 249 KB |
| `random-num100` | 0–100 | 350 | 638,208 | 1.23 MB | 277 KB |

Six quantizations per version: `F16`, `Q8_0`, `Q4_K_M` (recommended), `Q4_K_S`, `Q3_K_M`, `Q2_K`.

## Zero training

A 1-layer `llama` architecture whose only layer is all zeros. Weights were written by hand from a
closed-form construction — no dataset, no gradients, no GPU.

- Every attention / FFN weight is 0, so the layer is an identity.
- Every token embedding is one of exactly two one-hot vectors: "is a number" vs "is anything else".
  The hidden state therefore cannot depend on the input text or the context length.
- The output matrix gives all number tokens the same positive logit, so the softmax over the range is
  exactly uniform and the sampler performs a fair draw.
- EOS is the only positive logit once a number has been emitted, so generation stops after one token.

`random-digit` sampled 400 times from each of nine very different prompts (including a prompt injection
asking for a fixed digit) produced a uniform digit distribution every time: χ²(9) = 4.6 over 3,200 draws.

## Usage

```bash
# Ollama — import from this repo
ollama run hf.co/<user>/<this-repo>

# llama.cpp
llama-cli -m random-digit-Q4_K_M.gguf -p "hi" -n 4 --temp 1.0 --top-p 1.0 --top-k 10 -st
```

**Sampling parameters matter.** Use `top_p = 1.0` and `top_k ≥ number of values` (2 / 10 / 101).
With Ollama's defaults (`top_p=0.9`, `top_k=40`) the tail is always truncated, so some values become
impossible — for the 0–100 version only about 36 of 101 values remain reachable.

## Known quirks

- `prompt=""` returns an empty string (long-standing Ollama behaviour, unrelated to the model).
- In raw completion mode (no chat template), a prompt ending in a digit stops immediately, because
  "previous token is a number" is this model's stop signal. Chat mode is unaffected: the chat template
  is `{{ bos_token }}`, which discards the user text.

## Source

Generator, verification scripts, and the full derivation live in the GitHub repository:
`<replace with your GitHub URL>`.

## License

MIT.
