# Vocab / Tokenizer Analysis

## Dataset: fineweb10B_sp1024 — Train Set (1B tokens)

### Key stats

| Metric | Value |
|--------|-------|
| Total tokens | 1,000,000,000 |
| Unique token IDs used | 889 / 1024 (135 never fire) |
| Byte-fallback token usage | 7,928,615 (0.79%) |
| Compression | 0.41 tok/char (~2.4 chars/token) |
| Token entropy | 8.651 bits (max 9.796 for 889 tokens) |
| Vocab efficiency | 88.3% |

### Character distribution (decoded sample)

| Category | % |
|----------|----|
| Lowercase letters | 75.66% |
| Spaces | 16.86% |
| Uppercase letters | 3.44% |
| Punctuation | ~2.7% |
| Digits | 0.97% |
| Everything else | <0.5% |

660 distinct unicode characters, but ~270 of them appear only once across the entire dataset (CJK, Arabic, Hebrew, Devanagari, emojis, etc.).

---

## Problems

**1. 256 byte-fallback slots wasted**
Byte tokens (`0x00–0xFF`, IDs 5–260) exist as a lossless fallback for rare unicode. They fire only 0.79% of the time — 256 vocab slots doing almost no work.

**2. 135 token IDs never used**
Tokenizer was likely trained on slightly different data. Dead slots = wasted embedding parameters.

**3. Poor compression (0.41 tok/char)**
GPT-2 achieves ~0.25 tok/char. This tokenizer spends ~2.4 tokens per character, meaning the model sees less semantic content per training step.

**4. Non-ASCII is noise**
660 distinct chars but ~270 appear once in 1B tokens. These are scraping artifacts, not meaningful training signal.

---

## Recommendations

### 1. Retrain tokenizer without byte fallbacks ← highest ROI
```
--byte_fallback=false
```
Frees 256 slots for real BPE merges. Risk: ~0.79% of tokens become `<unk>` — acceptable noise given dataset scale.

### 2. Filter non-ASCII before tokenizing
Strip or normalize non-ASCII before training the tokenizer. Combined with (1), all 1024 slots go to English subwords.

### 3. Increase vocab to 2048 ← biggest quality win
Current compression (0.41 tok/char) is the main bottleneck. Doubling vocab would likely bring it closer to 0.3 tok/char — the model sees ~30% more semantic content per step for free.

Embedding cost at model_dim=512, tied, float32: **+2 MB** (4 MB total vs 2 MB now). Well within the 16 MB budget.

### 4. Combined: retrain at vocab=2048, no byte fallbacks, clean ASCII-only corpus
This is the ideal path. All 2048 slots go to useful English BPE pieces, compression improves significantly, no dead slots.

---

## Parameter cost reference (tied embeddings, float32)

| vocab_size | model_dim=512 |
|------------|--------------|
| 1024 (current) | 2.0 MB |
| 2048 | 4.0 MB |
| 4096 | 8.0 MB |
