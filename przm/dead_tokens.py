#!/usr/bin/env python3
"""Print token IDs that never appear in the val shard, with their piece text."""

from pathlib import Path
import numpy as np
import sentencepiece as spm

DATASET_DIR = Path("data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_1024_bpe.model")
MAGIC = 20240520


def read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype="<i4")
        assert header[0] == MAGIC
        n_toks = header[2]
        return np.frombuffer(f.read(n_toks * 2), dtype=np.uint16).copy()


sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
vocab_size = sp.get_piece_size()  # should be 1024

val_files = sorted(DATASET_DIR.glob("fineweb_val_*.bin"))
counts = np.zeros(vocab_size, dtype=np.int64)
for f in val_files:
    toks = read_bin(f)
    counts += np.bincount(toks, minlength=vocab_size)

dead = [i for i in range(vocab_size) if counts[i] == 0]
print(f"Vocab size : {vocab_size}")
print(f"Dead tokens: {len(dead)} / {vocab_size}\n")
print(f"{'id':>5}  {'piece':<30}  notes")
print("-" * 60)
for i in dead:
    piece = sp.id_to_piece(i)
    # annotate byte-fallback tokens (single raw bytes, piece looks like <0xNN>)
    note = "byte-fallback" if piece.startswith("<0x") else ""
    print(f"{i:>5}  {repr(piece):<30}  {note}")
