#!/usr/bin/env python3
"""
Disk-backed, chunked cosine distance analysis for k-mer count profiles.

This version is designed for larger genomic datasets where even a sparse
sample x vocabulary matrix may become too large. It avoids building any global
vocabulary or k-mer matrix in RAM.

How it works:
  1. Reads each FASTA file as a stream.
  2. Counts k-mers in small RAM chunks controlled by --chunk_size.
  3. Writes sorted partial count chunks to disk.
  4. Merges chunks into one sorted count file per sample.
  5. Computes exact pairwise cosine similarity by streaming two sorted count
     files at a time.
  6. Reads labels from a CSV file for silhouette analysis.

Example:
    python3 cosine-normal-chunked.py \
        --k 15 21 31 \
        --fasta_dir "./1-samples" \
        --out_dir "cosine_results" \
        --labels_csv "labels.csv" \
        --label_column group \
        --chunk_size 50000 \
        --reuse_matrix \
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
            "Chunked, disk-backed cosine distance analysis on genomic k-mer "
            "count profiles. Designed to avoid RAM crashes on larger datasets."
        )
    )

    parser.add_argument(
        "--fasta_dir",
        default="./1-samples",
        help="Directory containing FASTA files. Default: ./1-samples",
    )
    parser.add_argument(
        "--out_dir",
        default="cosine_results",
        help="Output directory. Default: cosine_results",
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
            "Maximum number of temporary chunk files opened at once during merge. "
            "Lower this if your OS has a strict open-file limit. Default: 24"
        ),
    )
    parser.add_argument(
        "--reuse_matrix",
        action="store_true",
        help=(
            "Reuse cached per-sample count files and cosine matrices when available. "
            "The name is kept for compatibility with your previous commands."
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

        # Support both exact filename/sample matches and extension-stripped matches.
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
# Chunked k-mer counting and sorted count files
# ============================================================

def cache_dirs(out_dir, k):
    base_dir = Path(out_dir) / "count_cache" / f"k{k}"
    counts_dir = base_dir / "counts"
    temp_dir = base_dir / "temp_chunks"
    matrix_dir = Path(out_dir) / "matrix_cache"

    counts_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, counts_dir, temp_dir, matrix_dir


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
        # Still rewrite it so we can compute final stats consistently.
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

                # Check on unique count, not total observations.
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

    # The merged stats are the final source of truth. The observation counters
    # are retained in metadata as useful diagnostics.
    meta = {
        "sample_name": sample_name,
        "fasta_path": str(fasta_path),
        "k": int(k),
        "count_file": str(count_file),
        "chunk_size": int(chunk_size),
        "n_temp_chunks_created": int(chunk_id),
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
    _, counts_dir, temp_dir, _ = cache_dirs(out_dir, k)

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
            "    unique k-mers: {unique:,} | total valid k-mers: {total:,} | chunks: {chunks}".format(
                unique=int(meta["unique_kmers"]),
                total=int(meta["total_valid_kmers"]),
                chunks=int(meta.get("n_temp_chunks_created", 0)),
            )
        )

    return sample_names, count_files, metadata


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


def cosine_matrix_from_count_files(count_files, metadata, matrix_cache_path=None, reuse_matrix=False):
    """Compute exact pairwise cosine similarity using sorted count files."""
    if reuse_matrix and matrix_cache_path and os.path.exists(matrix_cache_path):
        print(f"  Reusing cached cosine matrix: {matrix_cache_path}")
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

            # Numerical safety.
            sim = min(max(float(sim), 0.0), 1.0)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        print(f"  Pairwise cosine progress: {i + 1}/{n} rows completed")

    if matrix_cache_path:
        np.save(matrix_cache_path, sim_matrix)

    return sim_matrix


# ============================================================
# Silhouette and plotting
# ============================================================

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


def save_heatmap(dist_matrix, sample_names, k, out_dir):
    plt.figure(figsize=(9, 7))
    plt.imshow(dist_matrix, aspect="auto")
    plt.colorbar(label="Cosine distance")

    plt.xticks(range(len(sample_names)), sample_names, rotation=90)
    plt.yticks(range(len(sample_names)), sample_names)

    plt.title(f"Cosine Distance Matrix, k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cosine_distance_heatmap_k{k}.png"), dpi=300)
    plt.close()


def save_silhouette_barplot(sample_names, per_sample_silhouette, k, out_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(sample_names, per_sample_silhouette)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(rotation=90)
    plt.ylabel("Silhouette Score")
    plt.title(f"Per-sample Silhouette Scores Using Cosine Distance, k={k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cosine_per_sample_silhouette_k{k}.png"), dpi=300)
    plt.close()


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

    summary_results = []

    for k in args.k:
        print(f"\nProcessing k = {k}")

        _, _, _, matrix_cache_dir = cache_dirs(args.out_dir, k)
        cosine_cache_path = matrix_cache_dir / f"cosine_similarity_matrix_k{k}.npy"

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

        sim_matrix = cosine_matrix_from_count_files(
            count_files=count_files,
            metadata=metadata,
            matrix_cache_path=str(cosine_cache_path),
            reuse_matrix=args.reuse_matrix,
        )

        sim_matrix = np.asarray(sim_matrix, dtype=np.float64)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0

        dist_matrix = 1.0 - sim_matrix
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        # Save matrices.
        sim_df = pd.DataFrame(sim_matrix, index=sample_names, columns=sample_names)
        dist_df = pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names)

        sim_df.to_csv(os.path.join(args.out_dir, f"cosine_similarity_matrix_k{k}.csv"))
        dist_df.to_csv(os.path.join(args.out_dir, f"cosine_distance_matrix_k{k}.csv"))

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
            os.path.join(args.out_dir, f"cosine_per_sample_silhouette_k{k}.csv"),
            index=False,
        )

        vocabulary_size_estimate = int(sum(meta["unique_kmers"] for meta in metadata))
        total_valid_kmers = int(sum(meta["total_valid_kmers"] for meta in metadata))

        summary_results.append(
            {
                "method": "cosine_normal_chunked",
                "k": k,
                "n_samples": len(sample_names),
                "sum_unique_kmers_per_sample": vocabulary_size_estimate,
                "total_valid_kmers": total_valid_kmers,
                "chunk_size": args.chunk_size,
                "labels_csv": args.labels_csv,
                "sample_column": sample_column_used,
                "label_column": args.label_column,
                "n_classes": len(unique_labels),
                "classes": ";".join(map(str, unique_labels)),
                "global_silhouette_score": global_silhouette,
            }
        )

        print(f"  Sum of unique k-mers per sample: {vocabulary_size_estimate:,}")
        print(f"  Total valid k-mer observations: {total_valid_kmers:,}")
        if np.isnan(global_silhouette):
            print("  Global Silhouette Score: skipped")
        else:
            print(f"  Global Silhouette Score: {global_silhouette:.6f}")

        if not args.no_plots:
            save_heatmap(dist_matrix, sample_names, k, args.out_dir)
            save_silhouette_barplot(sample_names, per_sample_silhouette, k, args.out_dir)

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(args.out_dir, "cosine_global_results.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(summary_df)
    print(f"\nSaved global summary to: {summary_path}")


if __name__ == "__main__":
    main()