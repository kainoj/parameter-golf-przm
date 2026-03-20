#!/usr/bin/env python3
"""Analyze fineweb10B_sp1024 dataset statistics."""

import argparse
import numpy as np
import sentencepiece as spm
from collections import Counter
from pathlib import Path

DATASET_DIR = Path("data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_1024_bpe.model")
MAGIC = 20240520


def read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype="<i4")
    assert header[0] == MAGIC
    n_toks = header[2]
    with open(path, "rb") as f:
        f.seek(256 * 4)
        return np.frombuffer(f.read(n_toks * 2), dtype=np.uint16)


def analyze(toks: np.ndarray, sp: spm.SentencePieceProcessor, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    n = len(toks)
    print(f"\nTotal tokens : {n:>12,}")

    # token id stats
    counts = np.bincount(toks, minlength=1025)
    used_ids = np.count_nonzero(counts)
    print(f"Unique token IDs used : {used_ids} / 1024")

    # byte fallback tokens are IDs 5–260
    byte_tok_mask = np.zeros(1025, dtype=bool)
    byte_tok_mask[5:261] = True
    byte_tok_count = counts[5:261].sum()
    print(f"Byte-fallback tokens  : {byte_tok_count:>12,}  ({100*byte_tok_count/n:.2f}% of all tokens)")

    # EOS token usage (id=3)
    eos_count = int(counts[3])
    print(f"EOS tokens (<\\/s>)    : {eos_count:>12,}  (~{n//max(eos_count,1):,} tokens/doc avg doc length)")

    # top 20 tokens
    top20 = np.argsort(counts)[::-1][:20]
    print("\nTop 20 most frequent tokens:")
    print(f"  {'rank':>4}  {'id':>5}  {'count':>10}  {'%':>6}  text")
    for rank, tid in enumerate(top20, 1):
        text = repr(sp.id_to_piece(int(tid)))
        print(f"  {rank:>4}  {tid:>5}  {counts[tid]:>10,}  {100*counts[tid]/n:>5.2f}%  {text}")

    # decode a chunk and analyze unicode
    print("\nDecoding full token sequence for unicode analysis...")
    CHUNK = min(n, 5_000_000)
    text = sp.decode(toks[:CHUNK].tolist())
    total_chars = len(text)
    print(f"Decoded {CHUNK:,} tokens → {total_chars:,} chars  (compression: {CHUNK/total_chars:.2f} tok/char)")

    char_counts = Counter(text)
    distinct_chars = len(char_counts)
    print(f"Distinct unicode characters : {distinct_chars}")

    # unicode category breakdown
    import unicodedata
    cat_counts: Counter = Counter()
    for ch, cnt in char_counts.items():
        cat = unicodedata.category(ch)
        cat_counts[cat] += cnt
    total_ch = sum(cat_counts.values())
    cat_names = {
        "Lu": "Uppercase letter", "Ll": "Lowercase letter", "Lt": "Titlecase letter",
        "Lm": "Modifier letter",  "Lo": "Other letter",
        "Nd": "Decimal digit",    "Nl": "Letter number",   "No": "Other number",
        "Pc": "Connector punct",  "Pd": "Dash punct",      "Ps": "Open punct",
        "Pe": "Close punct",      "Pi": "Initial quot",    "Pf": "Final quot",
        "Po": "Other punct",      "Sm": "Math symbol",     "Sc": "Currency symbol",
        "Sk": "Modifier symbol",  "So": "Other symbol",
        "Zs": "Space separator",  "Zl": "Line separator",  "Zp": "Para separator",
        "Cc": "Control char",     "Cf": "Format char",
    }
    print("\nUnicode category breakdown:")
    for cat, cnt in cat_counts.most_common():
        name = cat_names.get(cat, cat)
        print(f"  {cat}  {name:<22}  {cnt:>10,}  ({100*cnt/total_ch:.2f}%)")

    # top 20 rarest chars → terminal
    print("\nTop 20 rarest characters:")
    all_sorted = sorted(char_counts.items(), key=lambda x: x[1])
    for ch, cnt in all_sorted[:20]:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "?"
        print(f"  U+{ord(ch):04X}  {repr(ch):<6}  count={cnt:<6}  {name}")

    # all chars sorted least → most frequent → file
    out_path = Path(f"char_freqs_{label.split()[0].lower()}.txt")
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"All {distinct_chars} characters (least → most frequent) — {label}\n")
        fout.write(f"  {'U+':>6}  {'repr':<8}  {'count':>10}  {'%':>6}  name\n")
        for ch, cnt in all_sorted:
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "?"
            fout.write(f"  U+{ord(ch):04X}  {repr(ch):<8}  {cnt:>10,}  {100*cnt/total_chars:>5.2f}%  {name}\n")
    print(f"Full character list written to {out_path}")

    # token entropy
    probs = counts[counts > 0] / n
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(used_ids)
    print(f"\nToken entropy : {entropy:.3f} bits  (max possible {max_entropy:.3f} bits for {used_ids} tokens)")
    print(f"Vocab efficiency : {100*entropy/max_entropy:.1f}%")


def load_concat(files: list[Path]) -> np.ndarray:
    return np.concatenate([read_bin(f) for f in files])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    val_files = sorted(DATASET_DIR.glob("fineweb_val_*.bin"))
    train_files = sorted(DATASET_DIR.glob("fineweb_train_*.bin"))

    print(f"Val files  : {len(val_files)}")
    print(f"Train files: {len(train_files)}")

    val_toks = load_concat(val_files)
    analyze(val_toks, sp, "VALIDATION SET")

    print("\nLoading train shards...")
    train_toks = load_concat(train_files)
    analyze(train_toks, sp, "TRAIN SET")


if __name__ == "__main__":
    main()
