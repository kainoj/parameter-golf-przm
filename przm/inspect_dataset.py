#!/usr/bin/env python3
"""Inspect fineweb10B_sp1024 .bin files."""

import argparse
import numpy as np
import sentencepiece as spm
from pathlib import Path

DATASET_DIR = Path("data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_1024_bpe.model")
MAGIC = 20240520


def read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype="<i4")
    assert header[0] == MAGIC, f"Bad magic: {header[0]}"
    n_toks = header[2]
    with open(path, "rb") as f:
        f.seek(256 * 4)
        toks = np.frombuffer(f.read(n_toks * 2), dtype=np.uint16)
    return toks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DATASET_DIR / "fineweb_val_000000.bin")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--n-tokens", type=int, default=200, help="tokens to decode and show")
    parser.add_argument("--offset", type=int, default=0, help="token offset to start from")
    parser.add_argument("--stats", action="store_true", help="show token frequency stats")
    parser.add_argument("--list-files", action="store_true", help="list all bin files with sizes")
    args = parser.parse_args()

    if args.list_files:
        files = sorted(DATASET_DIR.glob("*.bin"))
        total = 0
        for f in files:
            toks = read_bin(f)
            print(f"{f.name:40s}  {len(toks):>12,} tokens")
            total += len(toks)
        print(f"\nTotal: {total:,} tokens across {len(files)} files")
        return

    toks = read_bin(args.file)
    print(f"File: {args.file}")
    print(f"Tokens: {len(toks):,}  dtype={toks.dtype}  vocab_size=1024")

    if args.stats:
        counts = np.bincount(toks, minlength=1024)
        top = np.argsort(counts)[::-1][:20]
        print("\nTop 20 tokens by frequency:")
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        for tid in top:
            decoded = repr(sp.decode([int(tid)]))
            print(f"  id={tid:4d}  count={counts[tid]:>8,}  text={decoded}")
        return

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    chunk = toks[args.offset : args.offset + args.n_tokens]
    text = sp.decode(chunk.tolist())
    print(f"\n--- decoded tokens [{args.offset}:{args.offset + len(chunk)}] ---\n")
    print(text)


if __name__ == "__main__":
    main()
