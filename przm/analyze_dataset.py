#!/usr/bin/env python3
"""Analyze fineweb10B_sp1024 dataset statistics."""

import argparse
import os
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sentencepiece as spm

DATASET_DIR = Path("data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_1024_bpe.model")
MAGIC = 20240520
DECODE_CHUNK = 1_000_000  # tokens decoded at once per worker to bound peak memory

# sp loaded once per worker process via initializer
_worker_sp: spm.SentencePieceProcessor | None = None


def _init_worker(tokenizer_path: Path):
    global _worker_sp
    _worker_sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype="<i4")
        assert header[0] == MAGIC
        n_toks = header[2]
        return np.frombuffer(f.read(n_toks * 2), dtype=np.uint16).copy()


def _process_shard(path: Path) -> tuple:
    """Worker: runs in a subprocess. Returns (n_toks, bincount, char_counter)."""
    toks = read_bin(path)
    n = len(toks)
    bc = np.bincount(toks, minlength=1025).astype(np.int64)
    cc: Counter = Counter()
    for start in range(0, n, DECODE_CHUNK):
        cc.update(_worker_sp.decode(toks[start : start + DECODE_CHUNK].tolist()))
    return n, bc, cc


def analyze_streaming(files: list[Path], tokenizer_path: Path, sp: spm.SentencePieceProcessor, label: str, workers: int):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Shards: {len(files)}  Workers: {workers}", flush=True)

    total_n = 0
    counts = np.zeros(1025, dtype=np.int64)
    char_counts: Counter = Counter()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(tokenizer_path,),
    ) as pool:
        futs = {pool.submit(_process_shard, f): f for f in files}
        done = 0
        for fut in as_completed(futs):
            n, bc, cc = fut.result()
            total_n += n
            counts += bc
            char_counts += cc
            done += 1
            print(f"  [{done}/{len(files)}] {futs[fut].name}  total: {total_n:,}", flush=True)

    n = total_n
    print(f"\nTotal tokens : {n:>12,}")

    used_ids = int(np.count_nonzero(counts))
    print(f"Unique token IDs used : {used_ids} / 1024")

    byte_tok_count = int(counts[5:261].sum())
    print(f"Byte-fallback tokens  : {byte_tok_count:>12,}  ({100*byte_tok_count/n:.2f}% of all tokens)")

    eos_count = int(counts[3])
    print(f"EOS tokens (<\\/s>)    : {eos_count:>12,}  (~{n//max(eos_count,1):,} tokens/doc avg doc length)")

    top20 = np.argsort(counts)[::-1][:20]
    print("\nTop 20 most frequent tokens:")
    print(f"  {'rank':>4}  {'id':>5}  {'count':>10}  {'%':>6}  text")
    for rank, tid in enumerate(top20, 1):
        text = repr(sp.id_to_piece(int(tid)))
        print(f"  {rank:>4}  {tid:>5}  {counts[tid]:>10,}  {100*counts[tid]/n:>5.2f}%  {text}")

    total_chars = sum(char_counts.values())
    distinct_chars = len(char_counts)
    print(f"\nDecoded {n:,} tokens → {total_chars:,} chars  (compression: {n/max(total_chars,1):.2f} tok/char, all shards)")
    print(f"Distinct unicode characters : {distinct_chars}")

    cat_counts: Counter = Counter()
    for ch, cnt in char_counts.items():
        cat_counts[unicodedata.category(ch)] += cnt
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

    all_sorted = sorted(char_counts.items(), key=lambda x: x[1])
    print("\nTop 20 rarest characters:")
    for ch, cnt in all_sorted[:20]:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "?"
        print(f"  U+{ord(ch):04X}  {repr(ch):<6}  count={cnt:<6}  {name}")

    out_path = Path("przm") / f"char_freqs_{label.split()[0].lower()}.txt"
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

    nonzero = counts[counts > 0]
    probs = nonzero / n
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(used_ids))
    print(f"\nToken entropy : {entropy:.3f} bits  (max possible {max_entropy:.3f} bits for {used_ids} tokens)")
    print(f"Vocab efficiency : {100*entropy/max_entropy:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel worker processes (default: 32)")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    val_files = sorted(DATASET_DIR.glob("fineweb_val_*.bin"))
    train_files = sorted(DATASET_DIR.glob("fineweb_train_*.bin"))

    print(f"Val files  : {len(val_files)}")
    print(f"Train files: {len(train_files)}")

    w = args.workers
    analyze_streaming(val_files, args.tokenizer, sp, "VALIDATION SET", min(len(val_files), w))
    analyze_streaming(train_files, args.tokenizer, sp, "TRAIN SET", min(len(train_files), w))


if __name__ == "__main__":
    main()
