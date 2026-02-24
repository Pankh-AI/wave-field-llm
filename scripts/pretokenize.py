"""
Pre-tokenize FineWeb-Edu sample-10BT into memory-mapped binary shards.

Downloads the dataset via streaming, trains a 32K BPE tokenizer,
and writes shards as flat np.uint16 memmap files for zero-copy training.

Idempotent: skips already-completed shards (safe for spot preemption restarts).

Usage:
    python scripts/pretokenize.py --output-dir /data/tokenized
    python scripts/pretokenize.py --output-dir /data/tokenized --vocab-size 32000 --shard-size 10000000
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("pretokenize")


def setup_logging(output_dir: str):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(output_dir, "pretokenize.log")
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def train_bpe_tokenizer(dataset_iter, vocab_size: int, num_docs: int = 500_000):
    """Train byte-level BPE on streaming docs. Returns a tokenizers.Tokenizer."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    logger.info(f"Training BPE tokenizer (vocab={vocab_size}) on {num_docs:,} docs...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )

    def text_iterator():
        count = 0
        for item in dataset_iter:
            text = item["text"].strip()
            if len(text) > 50:
                yield text
                count += 1
                if count >= num_docs:
                    break
                if count % 100_000 == 0:
                    logger.info(f"  Tokenizer training: {count:,}/{num_docs:,} docs")

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    logger.info(f"  BPE vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def write_shard(tokens: list, shard_idx: int, output_dir: str) -> int:
    """Write a shard as flat uint16 memmap. Returns token count."""
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
    tmp_path = shard_path + ".tmp"

    arr = np.array(tokens, dtype=np.uint16)
    # Write to tmp then rename for atomicity
    mm = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm

    os.replace(tmp_path, shard_path)
    return len(arr)


def shard_is_complete(shard_idx: int, output_dir: str) -> bool:
    """Check if a shard file exists (not a .tmp partial)."""
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
    return os.path.exists(shard_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize FineWeb-Edu into binary shards")
    parser.add_argument("--output-dir", type=str, default="/data/tokenized",
                        help="Output directory for shards and tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000, help="BPE vocabulary size")
    parser.add_argument("--shard-size", type=int, default=10_000_000,
                        help="Tokens per shard (~20MB at uint16)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for manifest")
    parser.add_argument("--tokenizer-train-docs", type=int, default=500_000,
                        help="Number of docs to train tokenizer on")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT",
                        help="Dataset config name")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("  FineWeb-Edu Pre-tokenization")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {args.dataset} ({args.dataset_config})")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Vocab size: {args.vocab_size}")
    logger.info(f"  Shard size: {args.shard_size:,} tokens")

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset

    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")

    # Step 1: Train or load tokenizer
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        logger.info("Streaming dataset for tokenizer training...")
        tok_ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
        tokenizer = train_bpe_tokenizer(iter(tok_ds), args.vocab_size, args.tokenizer_train_docs)
        tokenizer.save(tokenizer_path)
        logger.info(f"  Tokenizer saved to {tokenizer_path}")

    actual_vocab = tokenizer.get_vocab_size()
    logger.info(f"  Vocab size: {actual_vocab}")

    # Step 2: Tokenize and write shards
    logger.info("\nTokenizing dataset into shards...")
    ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)

    shard_idx = 0
    current_tokens = []
    total_tokens = 0
    total_docs = 0
    skipped_shards = 0
    t0 = time.time()

    for item in ds:
        text = item["text"].strip()
        if len(text) < 50:
            continue

        ids = tokenizer.encode(text).ids
        current_tokens.extend(ids)
        total_docs += 1

        while len(current_tokens) >= args.shard_size:
            shard_tokens = current_tokens[:args.shard_size]
            current_tokens = current_tokens[args.shard_size:]

            if shard_is_complete(shard_idx, args.output_dir):
                # Already written — skip (idempotent)
                skipped_shards += 1
                shard_count = args.shard_size
            else:
                shard_count = write_shard(shard_tokens, shard_idx, args.output_dir)

            total_tokens += shard_count
            shard_idx += 1

            if shard_idx % 10 == 0:
                elapsed = time.time() - t0
                rate = total_tokens / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Shard {shard_idx:4d} | {total_tokens / 1e9:.2f}B tokens | "
                    f"{total_docs:,} docs | {rate / 1e6:.1f}M tok/s | "
                    f"skipped {skipped_shards}"
                )

    # Write final partial shard if any
    if len(current_tokens) > 0:
        if not shard_is_complete(shard_idx, args.output_dir):
            shard_count = write_shard(current_tokens, shard_idx, args.output_dir)
        else:
            shard_count = len(current_tokens)
            skipped_shards += 1
        total_tokens += shard_count
        shard_idx += 1

    elapsed = time.time() - t0

    # Step 3: Write manifest
    manifest = {
        "num_shards": shard_idx,
        "total_tokens": total_tokens,
        "vocab_size": actual_vocab,
        "seq_len": args.seq_len,
        "shard_size": args.shard_size,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "total_docs": total_docs,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("  Pre-tokenization complete")
    logger.info(f"  Shards: {shard_idx} ({skipped_shards} skipped/reused)")
    logger.info(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    logger.info(f"  Total docs: {total_docs:,}")
    logger.info(f"  Time: {elapsed / 3600:.1f}h")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
