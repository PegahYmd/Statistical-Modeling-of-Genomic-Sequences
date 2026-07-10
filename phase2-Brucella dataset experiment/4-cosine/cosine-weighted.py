#!/usr/bin/env python3
"""
Disk-backed, chunked weighted cosine analysis for genomic k-mer count profiles.

This version is designed to avoid RAM crashes on larger FASTA datasets.
It does NOT build a dense sample x vocabulary matrix in memory.

What it computes:
  1. raw cosine distance
  2. length-weighted cosine distance
  3. IDF / TF-IDF-style weighted cosine distance
  4. IDF + length-weighted cosine distance

The IDF weighting reduces the influence of very common k-mers and gives more
importance to k-mers that are more distinctive across samples. This is usually
more useful than only weighting by sequence length.

Example:
    python3 cosine-weighted-chunked.py \
        --k 15 \
        --fasta_dir "./1-samples" \
        --out_dir "cosine_weighted_chunked_results" \
        --labels_csv "labels.csv" \
        --label_column group \
        --chunk_size 10000 \
        --merge_fan_in 12 \
        --idf_max_df_fraction 0.90 \
        --no_plots
"""

import argparse
import glob
import heapq
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


# ============================================================
# Configuration helpers
# ============================================================

VALID_BASES = set("ACGT")
FASTA_EXTENSIONS = ("*.fa", "*.fasta", "*.fna", "*.fas")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chunked, disk-backed weighted cosine analysis on genomic k-mer "
            "count profiles. Designed to avoid RAM crashes."
        )
    )

    parser.add_argument(
        "--fasta_dir",
        default="./1-samples",
        help="Directory containing FASTA files. Default: ./1-samples",
    )
    parser.add_argument(
        "--out_dir",
        default="cosine_weighted_chunked_results",
        help="Output directory. Default: cosine_weighted_chunked_results",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[15, 21, 31],
        help="One or more k-mer lengths. Example: --k 15 21 31",
    )
    parser.add_argument(
        "--labels_csv",
        default="labels.csv",
        help="CSV file containing sample labels. Default: labels.csv",
    )
    parser.add_argument(
        "--label_column",
        default="group",
        help="Column in labels CSV used as class/group label. Default: group",
    )
    parser.add_argument(
        "--sample_column",
        default=None,
        help=(
            "Column in labels CSV containing sample names. If omitted, the script "
            "tries common names such as sample, sample_name, filename, file, id, name."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help=(
            "Maximum number of unique k-mers kept in RAM before writing a sorted "
            "chunk to disk. Smaller values use less RAM but run slower. Default: 50000"
        ),
    )
    parser.add_argument(
        "--merge_fan_in",
        type=int,
        default=24,
        help=(
            "Maximum number of temporary chunk/count files opened at once during merge. "
            "Lower this if your OS has a strict open-file limit. Default: 24"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "length", "idf", "idf_length"],
        choices=["raw", "length", "idf", "idf_length"],
        help=(
            "Methods to compute. Options: raw length idf idf_length. "
            "Default: all four."
        ),
    )
    parser.add_argument(
        "--idf_min_df",
        type=int,
        default=1,
        help=(
            "Minimum number of samples in which a k-mer must appear to be used in "
            "IDF cosine. Default: 1. Try 2 to remove singleton k-mers."
        ),
    )
    parser.add_argument(
        "--idf_max_df_fraction",
        type=float,
        default=1.0,
        help=(
            "Maximum document-frequency fraction allowed for IDF cosine. "
            "For example 0.90 removes k-mers present in more than 90%% of samples. "
            "Default: 1.0, keep all."
        ),
    )
    parser.add_argument(
        "--idf_power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to IDF weights. 1.0 is standard. Larger values emphasize "
            "distinctive k-mers more strongly. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--reuse_matrix",
        action="store_true",
        help=(
            "Reuse cached per-sample count files, IDF file, and cosine matrices when "
            "available. The name is kept for compatibility with previous commands."
        ),
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip heatmap and silhouette PNG plots to save time and memory.",
    )
    parser.add_argument(
        "--keep_temp_chunks",
        action="store_true",
        help="Keep temporary sorted chunk files for debugging. Default: delete them.",
    )

    args = parser.parse_args()

    if args.chunk_size < 1000:
        raise ValueError("--chunk_size is too small. Use at least 1000.")
    if args.merge_fan_in < 2:
        raise ValueError("--merge_fan_in must be at least 2.")
    if args.idf_min_df < 1:
        raise ValueError("--idf_min_df must be at least 1.")
    if not (0.0 < args.idf_max_df_fraction <= 1.0):
        raise ValueError("--idf_max_df_fraction must be in the range (0, 1].")
    if args.idf_power < 0:
        raise ValueError("--idf_power must be non-negative.")

    return args


def strip_fasta_suffix(name):
    """Return a stable sample name from a path or filename."""
    base = os.path.basename(str(name))

    for suffix in [
        ".fasta.gz", ".fa.gz", ".fna.gz", ".fas.gz",
        ".fasta", ".fa", ".fna", ".fas",
    ]:
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]

    return os.path.splitext(base)[0]


def safe_filename(name):
    """Make a safe filename while keeping it readable."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return safe if safe else "sample"


def find_fasta_files(fasta_dir):
    fasta_files = []
    for ext in FASTA_EXTENSIONS:
        fasta_files.extend(glob.glob(os.path.join(fasta_dir, ext)))

    fasta_files = sorted(set(fasta_files))

    if not fasta_files:
        raise FileNotFoundError(
            f"No FASTA files found in {fasta_dir}. "
            f"Expected extensions: {', '.join(FASTA_EXTENSIONS)}"
        )

    return fasta_files


# ============================================================
# Label loading from CSV
# ============================================================


def choose_sample_column(labels_df, label_column, sample_column=None):
    if sample_column is not None:
        if sample_column not in labels_df.columns:
            raise ValueError(
                f"Sample column '{sample_column}' was not found in labels CSV. "
                f"Available columns: {list(labels_df.columns)}"
            )
        return sample_column

    candidates = [
        "sample", "sample_name", "sample_id", "filename", "file",
        "name", "id", "genome", "accession", "strain",
    ]

    lower_to_original = {col.lower(): col for col in labels_df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    non_label_columns = [col for col in labels_df.columns if col != label_column]
    if not non_label_columns:
        raise ValueError(
            "Could not infer sample column from labels CSV. "
            "Please pass --sample_column explicitly."
        )

    return non_label_columns[0]


def load_labels_from_csv(labels_csv, sample_names, label_column, sample_column=None):
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_csv}. "
            "Use --labels_csv to point to your labels file."
        )

    labels_df = pd.read_csv(labels_csv)

    if label_column not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_column}' was not found in labels CSV. "
            f"Available columns: {list(labels_df.columns)}"
        )

    sample_column = choose_sample_column(labels_df, label_column, sample_column)

    label_lookup = {}
    for _, row in labels_df.iterrows():
        raw_name = str(row[sample_column])
        label = row[label_column]

        label_lookup[raw_name] = label
        label_lookup[strip_fasta_suffix(raw_name)] = label

    label_values = []
    missing = []

    for sample_name in sample_names:
        if sample_name in label_lookup:
            label_values.append(label_lookup[sample_name])
        else:
            missing.append(sample_name)

    if missing:
        raise ValueError(
            "The following FASTA samples were not found in the labels CSV: "
            + ", ".join(missing)
            + f"\nCSV sample column used: '{sample_column}'. "
            "Make sure filenames and CSV sample names match."
        )

    label_names = pd.Series(label_values, dtype="object").astype(str).to_numpy()
    label_codes, unique_labels = pd.factorize(label_names)

    print(f"  Loaded labels from: {labels_csv}")
    print(f"  Sample column used: {sample_column}")
    print(f"  Label column used for silhouette: {label_column}")
    print("  Label counts:")
    for label_name, count in pd.Series(label_names).value_counts().items():
        print(f"    {label_name}: {count}")

    return label_codes.astype(int), label_names, list(unique_labels), sample_column


# ============================================================
# Cache paths
# ============================================================


def cache_dirs(out_dir, k):
    base_dir = Path(out_dir) / "count_cache" / f"k{k}"
    counts_dir = base_dir / "counts"
    temp_dir = base_dir / "temp_chunks"
    matrix_dir = Path(out_dir) / "matrix_cache"
    weights_dir = Path(out_dir) / "weight_cache" / f"k{k}"

    counts_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, counts_dir, temp_dir, matrix_dir, weights_dir


def idf_cache_suffix(n_samples, min_df, max_df_fraction, idf_power):
    max_part = str(max_df_fraction).replace(".", "p")
    pow_part = str(idf_power).replace(".", "p")
    return f"n{n_samples}_mindf{min_df}_maxdf{max_part}_pow{pow_part}"


# ============================================================
# Sorted count file helpers
# ============================================================


def write_counter_chunk(counter, chunk_path):
    """Write one sorted Counter chunk to disk."""
    with open(chunk_path, "w", encoding="utf-8") as handle:
        for kmer, count in sorted(counter.items()):
            handle.write(f"{kmer}\t{count}\n")


def iter_count_file(path):
    """Yield (kmer, count) from a sorted count file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            kmer, count = line.split("\t")
            yield kmer, int(count)


def iter_weight_file(path):
    """Yield (kmer, weight_squared) from a sorted IDF weight file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            kmer, weight_sq = line.split("\t")
            yield kmer, float(weight_sq)


def merge_count_files_once(input_files, output_file, return_stats=False):
    """
    Merge sorted count files into one sorted count file.

    If return_stats is True, also compute total counts, unique k-mers, and
    squared L2 norm of the raw count vector.
    """
    iterators = [iter_count_file(path) for path in input_files]
    merged = heapq.merge(*iterators)

    total_count = 0
    unique_kmers = 0
    sumsq = 0

    current_kmer = None
    current_count = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for kmer, count in merged:
            if current_kmer is None:
                current_kmer = kmer
                current_count = count
            elif kmer == current_kmer:
                current_count += count
            else:
                out.write(f"{current_kmer}\t{current_count}\n")
                if return_stats:
                    total_count += current_count
                    unique_kmers += 1
                    sumsq += current_count * current_count
                current_kmer = kmer
                current_count = count

        if current_kmer is not None:
            out.write(f"{current_kmer}\t{current_count}\n")
            if return_stats:
                total_count += current_count
                unique_kmers += 1
                sumsq += current_count * current_count

    if return_stats:
        return {
            "total_valid_kmers": int(total_count),
            "unique_kmers": int(unique_kmers),
            "sumsq_counts": int(sumsq),
        }

    return None


def merge_count_files(input_files, final_output_file, temp_dir, fan_in=24, keep_temp=False):
    """Merge many sorted chunk files using limited open files."""
    input_files = [str(path) for path in input_files]

    if len(input_files) == 0:
        Path(final_output_file).write_text("", encoding="utf-8")
        return {"total_valid_kmers": 0, "unique_kmers": 0, "sumsq_counts": 0}

    if len(input_files) == 1:
        return merge_count_files_once(input_files, final_output_file, return_stats=True)

    round_id = 0
    files_to_merge = input_files
    created_intermediate = []

    while len(files_to_merge) > fan_in:
        next_round_files = []
        for group_id, start in enumerate(range(0, len(files_to_merge), fan_in)):
            group = files_to_merge[start : start + fan_in]
            intermediate = Path(temp_dir) / f"merge_round{round_id}_group{group_id}.tsv"
            merge_count_files_once(group, intermediate, return_stats=False)
            next_round_files.append(str(intermediate))
            created_intermediate.append(str(intermediate))
        files_to_merge = next_round_files
        round_id += 1

    stats = merge_count_files_once(files_to_merge, final_output_file, return_stats=True)

    if not keep_temp:
        all_temp = set(input_files + created_intermediate + files_to_merge)
        all_temp.discard(str(final_output_file))
        for path in all_temp:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    return stats


# ============================================================
# Chunked k-mer counting
# ============================================================


def count_sample_to_sorted_file(
    fasta_path,
    k,
    sample_name,
    counts_dir,
    temp_dir,
    chunk_size,
    merge_fan_in,
    reuse_counts=False,
    keep_temp_chunks=False,
):
    """Create one sorted k-mer count file for one FASTA sample."""
    safe_name = safe_filename(sample_name)
    count_file = Path(counts_dir) / f"{safe_name}.counts.tsv"
    meta_file = Path(counts_dir) / f"{safe_name}.metadata.json"
    sample_temp_dir = Path(temp_dir) / safe_name

    if reuse_counts and count_file.exists() and meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return str(count_file), meta

    if sample_temp_dir.exists():
        shutil.rmtree(sample_temp_dir)
    sample_temp_dir.mkdir(parents=True, exist_ok=True)

    counter = Counter()
    chunk_files = []
    chunk_id = 0
    tail = ""
    sequence_length_bases = 0
    valid_kmer_observations = 0
    skipped_kmer_observations = 0

    def flush_if_needed(force=False):
        nonlocal counter, chunk_id
        if not counter:
            return
        if force or len(counter) >= chunk_size:
            chunk_path = sample_temp_dir / f"chunk_{chunk_id:06d}.tsv"
            write_counter_chunk(counter, chunk_path)
            chunk_files.append(str(chunk_path))
            chunk_id += 1
            counter.clear()

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().upper()
            if not line or line.startswith(">"):
                continue

            sequence_length_bases += len(line)
            sequence = tail + line
            if len(sequence) < k:
                tail = sequence
                continue

            stop = len(sequence) - k + 1
            for i in range(stop):
                kmer = sequence[i : i + k]
                if all(base in VALID_BASES for base in kmer):
                    counter[kmer] += 1
                    valid_kmer_observations += 1
                else:
                    skipped_kmer_observations += 1

                if len(counter) >= chunk_size:
                    flush_if_needed(force=True)

            tail = sequence[-(k - 1) :] if k > 1 else ""

    flush_if_needed(force=True)

    stats = merge_count_files(
        input_files=chunk_files,
        final_output_file=count_file,
        temp_dir=sample_temp_dir,
        fan_in=merge_fan_in,
        keep_temp=keep_temp_chunks,
    )

    meta = {
        "sample_name": sample_name,
        "fasta_path": str(fasta_path),
        "k": int(k),
        "count_file": str(count_file),
        "chunk_size": int(chunk_size),
        "n_temp_chunks_created": int(chunk_id),
        "sequence_length_bases": int(sequence_length_bases),
        "valid_kmer_observations_seen": int(valid_kmer_observations),
        "skipped_kmer_observations_seen": int(skipped_kmer_observations),
        "total_valid_kmers": int(stats["total_valid_kmers"]),
        "unique_kmers": int(stats["unique_kmers"]),
        "sumsq_counts": int(stats["sumsq_counts"]),
    }

    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if not keep_temp_chunks:
        shutil.rmtree(sample_temp_dir, ignore_errors=True)

    return str(count_file), meta


def build_count_cache_for_k(
    fasta_files,
    k,
    out_dir,
    chunk_size,
    merge_fan_in,
    reuse_counts=False,
    keep_temp_chunks=False,
):
    """Build or reuse sorted per-sample count files for one k."""
    _, counts_dir, temp_dir, _, _ = cache_dirs(out_dir, k)

    sample_names = []
    count_files = []
    metadata = []

    for idx, fasta_path in enumerate(fasta_files):
        sample_name = strip_fasta_suffix(fasta_path)
        sample_names.append(sample_name)

        print(f"  Counting sample {idx + 1}/{len(fasta_files)}: {sample_name}")
        count_file, meta = count_sample_to_sorted_file(
            fasta_path=fasta_path,
            k=k,
            sample_name=sample_name,
            counts_dir=counts_dir,
            temp_dir=temp_dir,
            chunk_size=chunk_size,
            merge_fan_in=merge_fan_in,
            reuse_counts=reuse_counts,
            keep_temp_chunks=keep_temp_chunks,
        )

        count_files.append(count_file)
        metadata.append(meta)
        print(
            "    unique k-mers: {unique:,} | total valid k-mers: {total:,} | length: {length:,} | chunks: {chunks}".format(
                unique=int(meta["unique_kmers"]),
                total=int(meta["total_valid_kmers"]),
                length=int(meta.get("sequence_length_bases", 0)),
                chunks=int(meta.get("n_temp_chunks_created", 0)),
            )
        )

    return sample_names, count_files, metadata


# ============================================================
# IDF / TF-IDF weight file construction
# ============================================================


def iter_doc_presence_from_count_file(path):
    """Yield (kmer, 1) for each unique k-mer present in one sample count file."""
    for kmer, _count in iter_count_file(path):
        yield kmer, 1


def iter_df_file(path):
    """Yield (kmer, df) from a sorted document-frequency file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            kmer, df = line.split("\t")
            yield kmer, int(df)


def merge_to_df_once(input_files, output_df_file, source_mode):
    """
    Merge input files into a sorted document-frequency file.

    source_mode='count': each k-mer line contributes 1 sample occurrence.
    source_mode='df': each input value is already a partial df and is summed.
    """
    if source_mode == "count":
        iterators = [iter_doc_presence_from_count_file(path) for path in input_files]
    elif source_mode == "df":
        iterators = [iter_df_file(path) for path in input_files]
    else:
        raise ValueError("source_mode must be 'count' or 'df'.")

    merged = heapq.merge(*iterators)

    unique_kmers = 0
    current_kmer = None
    current_df = 0

    with open(output_df_file, "w", encoding="utf-8") as out:
        for kmer, df_part in merged:
            if current_kmer is None:
                current_kmer = kmer
                current_df = df_part
            elif kmer == current_kmer:
                current_df += df_part
            else:
                out.write(f"{current_kmer}\t{current_df}\n")
                unique_kmers += 1
                current_kmer = kmer
                current_df = df_part

        if current_kmer is not None:
            out.write(f"{current_kmer}\t{current_df}\n")
            unique_kmers += 1

    return unique_kmers


def convert_df_to_idf_weight_file(
    df_file,
    idf_file,
    n_samples,
    min_df,
    max_df_fraction,
    idf_power,
):
    """
    Convert document-frequency counts to a sorted weight-squared file.

    The cosine vector is count * weight, so pairwise dot and norm use weight^2.
    K-mers outside the df filter are skipped completely.
    """
    max_df = max(1, int(math.floor(max_df_fraction * n_samples)))

    kept = 0
    filtered_low = 0
    filtered_high = 0
    total = 0

    with open(idf_file, "w", encoding="utf-8") as out:
        for kmer, df in iter_df_file(df_file):
            total += 1

            if df < min_df:
                filtered_low += 1
                continue
            if df > max_df:
                filtered_high += 1
                continue

            # Smoothed IDF. Standard enough for TF-IDF-style weighting.
            idf = math.log((1.0 + n_samples) / (1.0 + df)) + 1.0
            weight = idf ** idf_power
            weight_sq = weight * weight

            if weight_sq <= 0.0:
                continue

            out.write(f"{kmer}\t{weight_sq:.17g}\n")
            kept += 1

    meta = {
        "n_samples": int(n_samples),
        "global_unique_kmers_before_filter": int(total),
        "global_unique_kmers_after_filter": int(kept),
        "idf_min_df": int(min_df),
        "idf_max_df_fraction": float(max_df_fraction),
        "idf_max_df_absolute": int(max_df),
        "idf_power": float(idf_power),
        "filtered_low_df": int(filtered_low),
        "filtered_high_df": int(filtered_high),
    }

    return meta


def build_idf_weight_file(
    count_files,
    n_samples,
    weights_dir,
    temp_dir,
    fan_in,
    min_df,
    max_df_fraction,
    idf_power,
    reuse=False,
):
    suffix = idf_cache_suffix(n_samples, min_df, max_df_fraction, idf_power)
    idf_file = Path(weights_dir) / f"idf_weights_{suffix}.tsv"
    meta_file = Path(weights_dir) / f"idf_weights_{suffix}.metadata.json"
    df_temp_dir = Path(temp_dir) / f"idf_df_{suffix}"

    if reuse and idf_file.exists() and meta_file.exists():
        print(f"  Reusing cached IDF weight file: {idf_file}")
        with open(meta_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return str(idf_file), meta, suffix

    if df_temp_dir.exists():
        shutil.rmtree(df_temp_dir)
    df_temp_dir.mkdir(parents=True, exist_ok=True)

    input_files = [str(path) for path in count_files]
    round_id = 0
    source_mode = "count"
    files_to_merge = input_files
    created_files = []

    while len(files_to_merge) > fan_in:
        next_files = []
        for group_id, start in enumerate(range(0, len(files_to_merge), fan_in)):
            group = files_to_merge[start : start + fan_in]
            out_df = df_temp_dir / f"df_round{round_id}_group{group_id}.tsv"
            merge_to_df_once(group, out_df, source_mode=source_mode)
            next_files.append(str(out_df))
            created_files.append(str(out_df))
        files_to_merge = next_files
        source_mode = "df"
        round_id += 1

    final_df = df_temp_dir / "global_document_frequency.tsv"
    merge_to_df_once(files_to_merge, final_df, source_mode=source_mode)

    meta = convert_df_to_idf_weight_file(
        df_file=final_df,
        idf_file=idf_file,
        n_samples=n_samples,
        min_df=min_df,
        max_df_fraction=max_df_fraction,
        idf_power=idf_power,
    )

    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    shutil.rmtree(df_temp_dir, ignore_errors=True)

    print(
        "  IDF weights built: kept {kept:,}/{total:,} global k-mers "
        "after df filtering".format(
            kept=int(meta["global_unique_kmers_after_filter"]),
            total=int(meta["global_unique_kmers_before_filter"]),
        )
    )

    return str(idf_file), meta, suffix


# ============================================================
# Pairwise cosine from sorted count files
# ============================================================


def dot_product_sorted_count_files(path_a, path_b):
    """Compute raw count dot product by streaming two sorted count files."""
    iter_a = iter_count_file(path_a)
    iter_b = iter_count_file(path_b)

    try:
        kmer_a, count_a = next(iter_a)
        kmer_b, count_b = next(iter_b)
    except StopIteration:
        return 0

    dot = 0

    while True:
        if kmer_a == kmer_b:
            dot += count_a * count_b
            try:
                kmer_a, count_a = next(iter_a)
                kmer_b, count_b = next(iter_b)
            except StopIteration:
                break
        elif kmer_a < kmer_b:
            try:
                kmer_a, count_a = next(iter_a)
            except StopIteration:
                break
        else:
            try:
                kmer_b, count_b = next(iter_b)
            except StopIteration:
                break

    return dot


def raw_cosine_matrix_from_count_files(count_files, metadata, matrix_cache_path=None, reuse_matrix=False):
    """Compute exact raw pairwise cosine similarity using sorted count files."""
    if reuse_matrix and matrix_cache_path and os.path.exists(matrix_cache_path):
        print(f"  Reusing cached raw cosine matrix: {matrix_cache_path}")
        return np.load(matrix_cache_path)

    n = len(count_files)
    sim_matrix = np.eye(n, dtype=np.float64)
    sumsq = np.array([float(meta["sumsq_counts"]) for meta in metadata], dtype=np.float64)

    for i in range(n):
        if sumsq[i] <= 0:
            sim_matrix[i, i] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if sumsq[i] <= 0 or sumsq[j] <= 0:
                sim = 0.0
            else:
                dot = dot_product_sorted_count_files(count_files[i], count_files[j])
                sim = dot / math.sqrt(sumsq[i] * sumsq[j])

            sim = min(max(float(sim), 0.0), 1.0)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        print(f"  Raw cosine progress: {i + 1}/{n} rows completed")

    if matrix_cache_path:
        np.save(matrix_cache_path, sim_matrix)

    return sim_matrix


def weighted_sumsq_count_file(count_file, weight_file):
    """Compute sum count^2 * weight^2 by streaming one count file and IDF file."""
    count_iter = iter_count_file(count_file)
    weight_iter = iter_weight_file(weight_file)

    try:
        kmer_c, count = next(count_iter)
        kmer_w, weight_sq = next(weight_iter)
    except StopIteration:
        return 0.0

    total = 0.0

    while True:
        if kmer_c == kmer_w:
            total += (count * count) * weight_sq
            try:
                kmer_c, count = next(count_iter)
            except StopIteration:
                break
            try:
                kmer_w, weight_sq = next(weight_iter)
            except StopIteration:
                break
        elif kmer_c < kmer_w:
            try:
                kmer_c, count = next(count_iter)
            except StopIteration:
                break
        else:
            try:
                kmer_w, weight_sq = next(weight_iter)
            except StopIteration:
                break

    return total


def weighted_dot_product_sorted_count_files(path_a, path_b, weight_file):
    """Compute sum count_a * count_b * weight^2 by streaming count/count/weight files."""
    iter_a = iter_count_file(path_a)
    iter_b = iter_count_file(path_b)
    iter_w = iter_weight_file(weight_file)

    try:
        kmer_a, count_a = next(iter_a)
        kmer_b, count_b = next(iter_b)
        kmer_w, weight_sq = next(iter_w)
    except StopIteration:
        return 0.0

    dot = 0.0

    while True:
        if kmer_a == kmer_b:
            target = kmer_a

            while kmer_w < target:
                try:
                    kmer_w, weight_sq = next(iter_w)
                except StopIteration:
                    return dot

            if kmer_w == target:
                dot += count_a * count_b * weight_sq

            try:
                kmer_a, count_a = next(iter_a)
                kmer_b, count_b = next(iter_b)
            except StopIteration:
                break
        elif kmer_a < kmer_b:
            try:
                kmer_a, count_a = next(iter_a)
            except StopIteration:
                break
        else:
            try:
                kmer_b, count_b = next(iter_b)
            except StopIteration:
                break

    return dot


def idf_cosine_matrix_from_count_files(
    count_files,
    weight_file,
    matrix_cache_path=None,
    reuse_matrix=False,
):
    """Compute exact IDF-weighted pairwise cosine similarity by streaming files."""
    if reuse_matrix and matrix_cache_path and os.path.exists(matrix_cache_path):
        print(f"  Reusing cached IDF cosine matrix: {matrix_cache_path}")
        return np.load(matrix_cache_path)

    n = len(count_files)
    sim_matrix = np.eye(n, dtype=np.float64)

    print("  Computing IDF-weighted norms")
    sumsq = np.zeros(n, dtype=np.float64)
    for i, count_file in enumerate(count_files):
        sumsq[i] = weighted_sumsq_count_file(count_file, weight_file)
        if sumsq[i] <= 0:
            sim_matrix[i, i] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if sumsq[i] <= 0 or sumsq[j] <= 0:
                sim = 0.0
            else:
                dot = weighted_dot_product_sorted_count_files(
                    count_files[i],
                    count_files[j],
                    weight_file,
                )
                sim = dot / math.sqrt(sumsq[i] * sumsq[j])

            sim = min(max(float(sim), 0.0), 1.0)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        print(f"  IDF cosine progress: {i + 1}/{n} rows completed")

    if matrix_cache_path:
        np.save(matrix_cache_path, sim_matrix)

    return sim_matrix


# ============================================================
# Length weight, silhouette, and plotting
# ============================================================


def compute_length_weight_matrix(sequence_lengths):
    """
    Computes a pairwise length-weight matrix.

    factor(i,j) = max(0, 1 - |Li - Lj| / ((Li + Lj)/2))
    """
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.float64)
    n = len(sequence_lengths)
    W = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            Li = sequence_lengths[i]
            Lj = sequence_lengths[j]
            mean_len = (Li + Lj) / 2.0
            if mean_len == 0:
                W[i, j] = 0.0
            else:
                penalty = abs(Li - Lj) / mean_len
                W[i, j] = max(0.0, 1.0 - penalty)

    return W


def similarity_to_distance(sim_matrix):
    sim_matrix = np.asarray(sim_matrix, dtype=np.float64)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
    dist_matrix = 1.0 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def compute_silhouette(dist_matrix, labels):
    n_samples = len(labels)
    n_classes = len(np.unique(labels))

    if n_classes < 2 or n_classes >= n_samples:
        print(
            "  Warning: silhouette score skipped because it requires at least "
            "2 classes and fewer classes than samples."
        )
        return np.nan, np.full(n_samples, np.nan)

    global_silhouette = silhouette_score(
        dist_matrix,
        labels,
        metric="precomputed",
    )

    per_sample_silhouette = silhouette_samples(
        dist_matrix,
        labels,
        metric="precomputed",
    )

    return global_silhouette, per_sample_silhouette


def save_heatmap(dist_matrix, sample_names, k, method, out_dir):
    plt.figure(figsize=(9, 7))
    plt.imshow(dist_matrix, aspect="auto")
    plt.colorbar(label=f"{method} cosine distance")

    plt.xticks(range(len(sample_names)), sample_names, rotation=90)
    plt.yticks(range(len(sample_names)), sample_names)

    plt.title(f"{method} Cosine Distance Matrix, k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{method}_cosine_distance_heatmap_k{k}.png"), dpi=300)
    plt.close()


def save_silhouette_barplot(sample_names, per_sample_silhouette, k, method, out_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(sample_names, per_sample_silhouette)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(rotation=90)
    plt.ylabel("Silhouette Score")
    plt.title(f"Per-sample Silhouette Scores, {method} Cosine, k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{method}_cosine_per_sample_silhouette_k{k}.png"), dpi=300)
    plt.close()


def save_method_outputs(
    method,
    k,
    sim_matrix,
    sample_names,
    labels,
    label_names,
    out_dir,
    no_plots,
):
    dist_matrix = similarity_to_distance(sim_matrix)

    sim_df = pd.DataFrame(sim_matrix, index=sample_names, columns=sample_names)
    dist_df = pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names)

    sim_df.to_csv(os.path.join(out_dir, f"{method}_cosine_similarity_matrix_k{k}.csv"))
    dist_df.to_csv(os.path.join(out_dir, f"{method}_cosine_distance_matrix_k{k}.csv"))

    global_silhouette, per_sample_silhouette = compute_silhouette(
        dist_matrix=dist_matrix,
        labels=labels,
    )

    per_sample_df = pd.DataFrame(
        {
            "sample": sample_names,
            "label_code": labels,
            "class": label_names,
            "silhouette_score": per_sample_silhouette,
        }
    )

    per_sample_df.to_csv(
        os.path.join(out_dir, f"{method}_cosine_per_sample_silhouette_k{k}.csv"),
        index=False,
    )

    if not no_plots:
        save_heatmap(dist_matrix, sample_names, k, method, out_dir)
        save_silhouette_barplot(sample_names, per_sample_silhouette, k, method, out_dir)

    return global_silhouette


# ============================================================
# Main analysis
# ============================================================


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fasta_files = find_fasta_files(args.fasta_dir)

    print(f"Found {len(fasta_files)} FASTA files in: {args.fasta_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Chunk size: {args.chunk_size:,} unique k-mers")
    print(f"Methods: {', '.join(args.methods)}")

    summary_results = []

    for k in args.k:
        print("\n" + "=" * 70)
        print(f"Processing k = {k}")
        print("=" * 70)

        _, _, temp_dir, matrix_cache_dir, weights_dir = cache_dirs(args.out_dir, k)
        raw_cache_path = matrix_cache_dir / f"raw_cosine_similarity_matrix_k{k}.npy"

        sample_names, count_files, metadata = build_count_cache_for_k(
            fasta_files=fasta_files,
            k=k,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
            merge_fan_in=args.merge_fan_in,
            reuse_counts=args.reuse_matrix,
            keep_temp_chunks=args.keep_temp_chunks,
        )

        labels, label_names, unique_labels, sample_column_used = load_labels_from_csv(
            labels_csv=args.labels_csv,
            sample_names=sample_names,
            label_column=args.label_column,
            sample_column=args.sample_column,
        )

        sequence_lengths = np.array(
            [int(meta.get("sequence_length_bases", 0)) for meta in metadata],
            dtype=np.int64,
        )

        length_df = pd.DataFrame(
            {
                "sample": sample_names,
                "class": label_names,
                "sequence_length": sequence_lengths,
                "unique_kmers": [int(meta["unique_kmers"]) for meta in metadata],
                "total_valid_kmers": [int(meta["total_valid_kmers"]) for meta in metadata],
            }
        )
        length_df.to_csv(os.path.join(args.out_dir, f"sequence_lengths_k{k}.csv"), index=False)

        raw_sim = None
        length_weight_matrix = None
        idf_sim = None
        idf_meta = {}

        if "raw" in args.methods or "length" in args.methods:
            raw_sim = raw_cosine_matrix_from_count_files(
                count_files=count_files,
                metadata=metadata,
                matrix_cache_path=str(raw_cache_path),
                reuse_matrix=args.reuse_matrix,
            )

        if "length" in args.methods or "idf_length" in args.methods:
            length_weight_matrix = compute_length_weight_matrix(sequence_lengths)
            pd.DataFrame(
                length_weight_matrix,
                index=sample_names,
                columns=sample_names,
            ).to_csv(os.path.join(args.out_dir, f"length_weight_matrix_k{k}.csv"))

        if "idf" in args.methods or "idf_length" in args.methods:
            idf_file, idf_meta, idf_suffix = build_idf_weight_file(
                count_files=count_files,
                n_samples=len(sample_names),
                weights_dir=weights_dir,
                temp_dir=temp_dir,
                fan_in=args.merge_fan_in,
                min_df=args.idf_min_df,
                max_df_fraction=args.idf_max_df_fraction,
                idf_power=args.idf_power,
                reuse=args.reuse_matrix,
            )

            idf_cache_path = matrix_cache_dir / f"idf_cosine_similarity_matrix_k{k}_{idf_suffix}.npy"
            idf_sim = idf_cosine_matrix_from_count_files(
                count_files=count_files,
                weight_file=idf_file,
                matrix_cache_path=str(idf_cache_path),
                reuse_matrix=args.reuse_matrix,
            )

        method_to_sim = {}
        if "raw" in args.methods:
            method_to_sim["raw"] = raw_sim
        if "length" in args.methods:
            method_to_sim["length_weighted"] = raw_sim * length_weight_matrix
        if "idf" in args.methods:
            method_to_sim["idf_weighted"] = idf_sim
        if "idf_length" in args.methods:
            method_to_sim["idf_length_weighted"] = idf_sim * length_weight_matrix

        vocabulary_size_estimate = int(sum(meta["unique_kmers"] for meta in metadata))
        total_valid_kmers = int(sum(meta["total_valid_kmers"] for meta in metadata))
        global_unique_after_filter = idf_meta.get("global_unique_kmers_after_filter", np.nan)
        global_unique_before_filter = idf_meta.get("global_unique_kmers_before_filter", np.nan)

        for method, sim_matrix in method_to_sim.items():
            print(f"\n  Saving/evaluating method: {method}")
            global_silhouette = save_method_outputs(
                method=method,
                k=k,
                sim_matrix=sim_matrix,
                sample_names=sample_names,
                labels=labels,
                label_names=label_names,
                out_dir=args.out_dir,
                no_plots=args.no_plots,
            )

            summary_results.append(
                {
                    "method": method,
                    "k": k,
                    "n_samples": len(sample_names),
                    "sum_unique_kmers_per_sample": vocabulary_size_estimate,
                    "total_valid_kmers": total_valid_kmers,
                    "min_sequence_length": int(np.min(sequence_lengths)),
                    "max_sequence_length": int(np.max(sequence_lengths)),
                    "mean_sequence_length": float(np.mean(sequence_lengths)),
                    "chunk_size": args.chunk_size,
                    "labels_csv": args.labels_csv,
                    "sample_column": sample_column_used,
                    "label_column": args.label_column,
                    "n_classes": len(unique_labels),
                    "classes": ";".join(map(str, unique_labels)),
                    "idf_min_df": args.idf_min_df,
                    "idf_max_df_fraction": args.idf_max_df_fraction,
                    "idf_power": args.idf_power,
                    "global_unique_kmers_before_idf_filter": global_unique_before_filter,
                    "global_unique_kmers_after_idf_filter": global_unique_after_filter,
                    "global_silhouette_score": global_silhouette,
                }
            )

            if np.isnan(global_silhouette):
                print(f"  {method} silhouette: skipped")
            else:
                print(f"  {method} silhouette: {global_silhouette:.6f}")

        print(f"\n  Sum of unique k-mers per sample: {vocabulary_size_estimate:,}")
        print(f"  Total valid k-mer observations: {total_valid_kmers:,}")

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(args.out_dir, "cosine_weighted_global_results.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(summary_df)
    print(f"\nSaved global summary to: {summary_path}")


if __name__ == "__main__":
    main()