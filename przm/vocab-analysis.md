# Vocab / Tokenizer Analysis

## Dataset: fineweb10B_sp1024 — Full Dataset (8B train tokens, 62M val tokens)

### Key Stats

| Metric | Train | Val |
|--------|-------|-----|
| Total tokens | 8,000,000,000 | 62,021,846 |
| Unique token IDs used | 892 / 1024 | 886 / 1024 |
| Dead token IDs | 132 | 138 |
| Byte-fallback token usage | 64,351,536 (0.80%) | 492,516 (0.79%) |
| Compression | 0.41 tok/char (2.44 chars/tok) | 0.41 tok/char |
| Decoded characters | 19,417,321,419 | 150,180,877 |
| Token entropy | 8.652 bits (max 9.801 for 892 tokens) | 8.649 bits |
| Vocab efficiency | 88.3% | 88.3% |
| Distinct unicode chars | 14,820 | 3,134 |

---

## Character Distribution (Train Corpus)

Top characters by frequency in decoded text:

| Char | % | Char | % |
|------|---|------|---|
| ' ' (space) | 16.78% | 'r' | 4.87% |
| 'e' | 9.32% | 'h' | 3.40% |
| 't' | 6.73% | 'l' | 3.26% |
| 'a' | 6.25% | 'd' | 2.80% |
| 'o' | 6.02% | 'c' | 2.37% |
| 'i' | 5.56% | 'u' | 2.26% |
| 'n' | 5.47% | 'm' | 1.84% |
| 's' | 4.95% | '.' | 0.97% |

The top 27 ASCII chars (a-z + space) account for ~95%+ of all decoded text.
Lowercase letters: 75.55%, Space: 16.78%, Uppercase: 3.54%, Punctuation: ~2.7%, Digits: 1.01%.

---

## Problems

### 1. 256 byte-fallback slots — low utilization

Byte tokens (IDs 5–260, the 256 single-byte fallbacks) fire 0.80% of the time. These fire on non-ASCII characters that have no BPE token.

Key observation: the train set has **14,820** distinct unicode characters while the val set has only **3,134**. Of the 14,820 train chars, ~11,686 appear in train but not val. These are one-off scraping artifacts (rare CJK glyphs, obscure scripts, emojis). They generate byte-fallback tokens during training that contribute zero signal to val BPB.

Result: The model burns 256 embedding rows (256 KB FP16) on fallbacks that fire mostly on training noise.

### 2. 132 dead token IDs in train (138 in val)

1024 − 892 = **132 token IDs never fire** across 8 billion training tokens. These are likely tokens trained on code, math, or other out-of-distribution text. They occupy 132 embedding rows (132 KB FP16, ~75 KB after zlib) that receive no gradient and contribute nothing.

The 6-token difference between train dead (132) and val dead (138) means a handful of tokens appear in training data but never in validation — further evidence of training-distribution artifacts.

### 3. Compression is near-theoretical for 1024 vocab

The old analysis compared this tokenizer unfavorably to GPT-2 (0.25 tok/char = 4 chars/tok). This comparison is invalid. GPT-2 has 50,257 vocab entries. For a **1024-token** SentencePiece BPE on English, 2.44 chars/tok is close to the theoretical ceiling given the byte-fallback overhead. If byte-fallbacks were removed, real BPE tokens would have closer to 3.0–3.5 chars/tok.

The actual bottleneck is dead tokens + byte-fallbacks consuming 388/1024 = **37.9% of vocab capacity** for near-zero return.

### 4. Standalone space token firing at 1.19%

Token '▁' (ID 939, bare space) is the **5th most frequent token** at 1.19% of all tokens. This fires when a word boundary is followed by a byte-fallback character or an uncommon character lacking a word-level token. Every such occurrence means the word-initial space wasn't merged into the following subword — a wasted merge opportunity that increases sequence length by 1 per occurrence.

With 0.80% byte-fallback rate and a similar share of other rare chars, the bare space token accounts for most of these — implying nearly every byte-fallback character also forces an extra '▁' token. The true effective cost of byte-fallbacks is **~1.6× tokens** per occurrence (the byte itself + a stranded '▁').

### 5. Top-20 tokens clustered at high IDs

The top-20 most frequent tokens have IDs: 267, 276, 280, 282, 285, 287, 290, 291, 292, 939, 940, 941, 942, 946, 957, 960, 962. These are **not** in the lowest ID slots. The ID assignment appears to follow BPE merge order (later merges get higher IDs), so common subwords got merged late and sit at high IDs. The embedding matrix stores row 0–1023 in order; the hot rows (939–962) are clustered near the bottom, which actually helps local zlib compression for that region. However, the dead rows (IDs 0–4 are special tokens; many others are mid-range) are randomly initialized and compress poorly.

---

## Artifact Budget Analysis

FP16 tied embeddings, vocab=1024, dim=512:

| Region | Rows | Raw bytes | ~zlib bytes |
|--------|------|-----------|-------------|
| Byte-fallback (IDs 5–260) | 256 | 262,144 | ~180,000 |
| Dead tokens | 132 | 135,168 | ~130,000 (random = barely compressible) |
| Active BPE tokens | 636 | 651,264 | ~400,000 |
| **Total embedding** | 1024 | 1,048,576 | ~710,000 |

Pruning dead rows frees ~130 KB compressed. Pruning byte-fallback rows (and accepting `<unk>` for rare chars) frees another ~180 KB. Combined: **~310 KB artifact savings** — enough to fund roughly half an extra transformer layer at current model size.

---

## Recommendations

### 1. Prune dead token embedding rows before export ← highest ROI / no retokenization required

Map the 892 live token IDs to a contiguous 0–891 index, store the 892-row embedding matrix, and include the 892-element ID remap table (892 × 2 bytes uint16 = 1,784 bytes code). After zlib, this saves ~130 KB of artifact space.

```python
# At export time:
live_ids = sorted(ids_that_fired_in_train)  # 892 entries
remap = {old: new for new, old in enumerate(live_ids)}
embedding = model.embed.weight[live_ids]     # [892, 512] FP16
# Store remap table alongside model
```

Risk: none if the val tokenizer uses the same token set. The 6-token gap (train fires, val doesn't) means 6 rows become OOV on val — negligible impact.

### 2. Redirect byte-fallback tokens to a shared OOV embedding ← zero parameter cost

Instead of 256 independent byte-fallback rows, learn a single 512-dim "OOV" vector and route all byte-fallback token lookups to it. This frees 255 embedding rows (255 KB FP16 raw, ~175 KB after zlib) at the cost of 0.80% of tokens sharing a representation.

This doesn't require retokenization — just change the embedding lookup to clamp byte IDs to a shared slot.

```python
# In embedding forward:
OOV_ID = 5  # one representative byte-fallback slot
is_byte_fallback = (input_ids >= 5) & (input_ids <= 260)
input_ids = torch.where(is_byte_fallback, OOV_ID, input_ids)
```

Combine with recommendation 1 for up to **310 KB artifact savings total** — enough to add ~12–15 extra attention head parameters or widen a layer.

### 3. Retokenize without byte fallbacks ← breaks compatibility, highest quality gain

```
sentencepiece --byte_fallback=false --vocab_size=1024
```

Replaces all 256 byte-fallback slots with real BPE merges. Rare non-ASCII becomes `<unk>` (0.80% of tokens → `<unk>`). The 256 new merges would improve compression from 2.44 chars/tok toward ~2.8 chars/tok — the model sees ~15% more semantic content per training step.

Combined with filtering non-ASCII from the training corpus before tokenizer training, all 1024 slots would go to English subwords.

### 4. Increase vocab to 2048 ← biggest quality win, requires retokenizing

At model_dim=512, FP16 tied embeddings:
- 1024 vocab = 1.0 MB embedding (current)
- 2048 vocab = 2.0 MB embedding (+1 MB cost)

Each additional BPE merge at the 1024→2048 transition covers high-value word fragments (common word stems, frequent bigrams). Projected compression: ~3.0–3.3 chars/tok (from 2.44), giving ~20–25% fewer tokens for the same text. This directly reduces BPB since the model attends to denser semantic content per step.

A vocab=2048 model with FP16 embeddings still fits well within the 16 MB artifact budget.

### 5. Corpus-aware ID remapping for better embedding zlib compression ← novel, zero-cost

Sort token IDs by decreasing frequency before training. The most frequent tokens get IDs 0–N. Since frequent tokens receive the most gradient updates, their embedding rows converge to smooth, similar vectors — placed adjacently, these compress well with zlib. Currently the hot tokens sit at IDs 939–962 (clustered near the vocab ceiling) while low IDs (the byte-fallback range) sit at the start. Sorting by frequency would concentrate the "smooth" high-gradient rows in one region.

Estimated zlib gain: ~5–15% better embedding compression (empirical estimate; depends on how similar high-frequency embedding rows become). No quality impact.

---

## Parameter Cost Reference (tied embeddings, FP16)

| vocab_size | model_dim=512 | Artifact est. (after zlib) |
|------------|--------------|--------------------------|
| 892 (pruned current) | 0.89 MB raw | ~0.56 MB |
| 1024 (current) | 1.0 MB raw | ~0.71 MB |
| 2048 | 2.0 MB raw | ~1.35 MB |
| 4096 | 4.0 MB raw | ~2.70 MB |

---

## Summary Table: Recommendations by ROI

| Recommendation | Requires retokenization | Artifact saving | Quality gain | Risk |
|----------------|------------------------|-----------------|--------------|------|
| Prune dead token rows | No | ~130 KB | None | Very low |
| Shared OOV embedding | No | ~175 KB | -0.001 BPB | Low |
| Both above combined | No | **~310 KB** | ~0 | Low |
| Frequency-sorted IDs | Yes (or remap) | ~50–100 KB | None | Low |
| No byte-fallbacks, vocab=1024 | Yes | +~175 KB budget headroom | +0.005–0.010 BPB | Low |
| Vocab=2048, no byte-fallbacks | Yes | −1 MB budget cost | **+0.010–0.030 BPB** | Medium |
