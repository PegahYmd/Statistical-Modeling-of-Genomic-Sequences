#!/usr/bin/env python3

import os
import re
import gzip
import heapq
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import MDS


# ============================================================
# FILE / SAMPLE UTILITIES
# ============================================================

FASTA_EXTENSIONS = (
    ".fa", ".fasta", ".fna",
    ".fa.gz", ".fasta.gz", ".fna.gz"
)

_RC_TABLE = str.maketrans("ACGT", "TGCA")


def open_text_auto(path, mode="rt"):
    """Open normal text files and .gz files transparently."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def sample_name_from_path(path):
    """Return filename without FASTA extension."""
    name = os.path.basename(path)
    name = re.sub(r"\.(fa|fasta|fna)(\.gz)?$", "", name, flags=re.IGNORECASE)
    return name


def safe_filename(name):
    """Make a sample name safe for output filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def normalize_sample_key(value):
    """Normalize sample names for matching FASTA names to labels CSV rows."""
    value = str(value).strip()
    value = os.path.basename(value)
    value = re.sub(r"\.(fa|fasta|fna)(\.gz)?$", "", value, flags=re.IGNORECASE)
    return value.lower()


# ============================================================
# k-mer UTILITIES
# ============================================================

def reverse_complement(seq):
    return seq.translate(_RC_TABLE)[::-1]


def canonical_kmer(kmer):
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def count_kmers_streaming(path, k, canonical=True):
    """
    Count k-mers from a FASTA file without loading the whole sequence.

    Important:
    - Reads line by line.
    - Does not create k-mers across different FASTA records/contigs.
    - Ignores k-mers containing bases outside A/C/G/T.
    - Stores only the current sample's Counter in memory.
    """
    counts = Counter()
    total_bp = 0
    valid_kmers = 0
    tail = ""

    with open_text_auto(path, "rt") as f:
        for line in f:
            line = line.strip().upper()

            if not line:
                continue

            if line.startswith(">"):
                tail = ""
                continue

            total_bp += len(line)
            chunk = tail + line

            if len(chunk) >= k:
                end = len(chunk) - k + 1

                for i in range(end):
                    kmer = chunk[i:i + k]

                    # Faster than regex for very large datasets
                    if any(base not in "ACGT" for base in kmer):
                        continue

                    if canonical:
                        kmer = canonical_kmer(kmer)

                    counts[kmer] += 1
                    valid_kmers += 1

                tail = chunk[-(k - 1):] if k > 1 else ""
            else:
                tail = chunk

    return counts, total_bp, valid_kmers


def write_sorted_count_file(counts, output_path):
    """Write one sample's k-mer counts as sorted TSV.GZ: kmer<TAB>count."""
    with gzip.open(output_path, "wt") as out:
        for kmer, count in sorted(counts.items()):
            out.write(f"{kmer}\t{count}\n")


def read_count_line(handle):
    line = handle.readline()
    if not line:
        return None

    kmer, count = line.rstrip("\n").split("\t")
    return kmer, int(count)


# ============================================================
# LABEL HANDLING
# ============================================================

def infer_label(filename):
    """
    Fallback label inference if no labels CSV is supplied or a sample is missing.

    Edit this function only if you still want filename-based fallback rules.
    """
    name = filename.lower()

    if "functional" in name:
        return "functional"

    if "pathological" in name or "pathogenic" in name:
        return "pathological"

    match = re.search(r"sample(\d+)", name)
    if match:
        num = int(match.group(1))

        if 1 <= num <= 5:
            return "functional"

        if 6 <= num <= 10:
            return "pathological"

    return "unknown"


def choose_column(df, requested, candidates, role):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested {role} column '{requested}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    lower_to_original = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(
        f"Could not automatically choose the {role} column. "
        f"Available columns: {list(df.columns)}. "
        f"Please provide --{role}_column explicitly."
    )


def load_labels_from_csv(labels_csv, label_column=None, sample_column=None):
    """Load labels from CSV and return a normalized sample -> label dictionary."""
    df = pd.read_csv(labels_csv)

    chosen_label_col = choose_column(
        df,
        label_column,
        candidates=["group", "label", "class", "category", "species", "type"],
        role="label"
    )

    if sample_column is None:
        sample_candidates = [
            "sample", "sample_name", "sample_id", "filename", "file",
            "name", "id", "accession", "genome", "isolate"
        ]

        lower_to_original = {c.lower(): c for c in df.columns}
        chosen_sample_col = None

        for candidate in sample_candidates:
            if candidate.lower() in lower_to_original:
                chosen_sample_col = lower_to_original[candidate.lower()]
                break

        if chosen_sample_col is None:
            non_label_cols = [c for c in df.columns if c != chosen_label_col]
            if not non_label_cols:
                raise ValueError(
                    "The labels CSV must contain a sample column and a label column."
                )
            chosen_sample_col = non_label_cols[0]
    else:
        if sample_column not in df.columns:
            raise ValueError(
                f"Requested sample column '{sample_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        chosen_sample_col = sample_column

    label_map = {}

    for _, row in df.iterrows():
        sample_value = row[chosen_sample_col]
        label_value = row[chosen_label_col]

        if pd.isna(sample_value) or pd.isna(label_value):
            continue

        key = normalize_sample_key(sample_value)
        label_map[key] = str(label_value).strip()

    return label_map, chosen_sample_col, chosen_label_col


def get_group_labels(fasta_files, labels_csv=None, label_column=None, sample_column=None):
    sample_names = [sample_name_from_path(path) for path in fasta_files]

    if labels_csv is None:
        print("\nNo labels CSV supplied. Using fallback filename-based label inference.")
        group_labels = [infer_label(os.path.basename(path)) for path in fasta_files]
        return sample_names, group_labels, None, None

    label_map, used_sample_column, used_label_column = load_labels_from_csv(
        labels_csv=labels_csv,
        label_column=label_column,
        sample_column=sample_column
    )

    print(f"\nLoaded labels from: {labels_csv}")
    print(f"Sample column used: {used_sample_column}")
    print(f"Label column used for silhouette: {used_label_column}")

    group_labels = []
    missing = []

    for path, sample in zip(fasta_files, sample_names):
        candidates = [
            normalize_sample_key(sample),
            normalize_sample_key(os.path.basename(path)),
            normalize_sample_key(path),
        ]

        label = None
        for key in candidates:
            if key in label_map:
                label = label_map[key]
                break

        if label is None:
            label = "unknown"
            missing.append(sample)

        group_labels.append(label)

    if missing:
        print("\nWarning: labels were not found for these samples:")
        for sample in missing:
            print(f"  {sample}")

    return sample_names, group_labels, used_sample_column, used_label_column


# ============================================================
# kWIP ENTROPY WEIGHTING
# ============================================================

def binary_entropy_scalar(F):
    """Binary entropy H(F) with base-2 logarithm."""
    if F <= 0.0 or F >= 1.0:
        return 0.0
    return -F * np.log2(F) - (1.0 - F) * np.log2(1.0 - F)


def entropy_summary_from_document_frequency(document_frequency, n_samples):
    H_values = np.fromiter(
        (binary_entropy_scalar(df_count / n_samples)
         for df_count in document_frequency.values()),
        dtype=float,
        count=len(document_frequency)
    )

    if H_values.size == 0:
        return {
            "min_entropy_weight": np.nan,
            "max_entropy_weight": np.nan,
            "mean_entropy_weight": np.nan,
            "median_entropy_weight": np.nan,
            "zero_weight_kmers": 0,
            "nonzero_weight_kmers": 0,
            "H_values": H_values
        }

    return {
        "min_entropy_weight": float(np.min(H_values)),
        "max_entropy_weight": float(np.max(H_values)),
        "mean_entropy_weight": float(np.mean(H_values)),
        "median_entropy_weight": float(np.median(H_values)),
        "zero_weight_kmers": int(np.sum(H_values == 0)),
        "nonzero_weight_kmers": int(np.sum(H_values > 0)),
        "H_values": H_values
    }


# ============================================================
# MEMORY-SAFE kWIP COMPUTATION
# ============================================================

def build_count_files(fasta_files, k, output_dir, canonical=True, reuse_counts=False):
    """
    First pass:
    - Count k-mers sample by sample.
    - Save each sample's sorted counts to disk.
    - Build document frequency df(x): number of samples containing k-mer x.

    This avoids storing all sample Counters or a dense count matrix in RAM.
    """
    counts_dir = os.path.join(output_dir, "kmer_counts")
    os.makedirs(counts_dir, exist_ok=True)

    document_frequency = Counter()
    count_files = []
    sample_stats = []

    for sample_idx, path in enumerate(fasta_files):
        sample = sample_name_from_path(path)
        count_path = os.path.join(
            counts_dir,
            f"{sample_idx:05d}_{safe_filename(sample)}_k{k}_counts.tsv.gz"
        )
        count_files.append(count_path)

        if reuse_counts and os.path.exists(count_path):
            distinct_kmers = 0
            total_counted_kmers = 0
            with gzip.open(count_path, "rt") as f:
                for line in f:
                    kmer, count = line.rstrip("\n").split("\t")
                    document_frequency[kmer] += 1
                    distinct_kmers += 1
                    total_counted_kmers += int(count)

            sample_stats.append({
                "sample": sample,
                "file": os.path.basename(path),
                "total_bp": np.nan,
                "valid_counted_kmers": total_counted_kmers,
                "distinct_kmers": distinct_kmers,
                "count_file": count_path,
                "reused_count_file": True
            })

            print(
                f"Reused {os.path.basename(count_path)}: "
                f"{distinct_kmers:,} distinct k-mers"
            )
            continue

        counts, total_bp, valid_kmers = count_kmers_streaming(
            path=path,
            k=k,
            canonical=canonical
        )

        for kmer in counts.keys():
            document_frequency[kmer] += 1

        write_sorted_count_file(counts, count_path)

        sample_stats.append({
            "sample": sample,
            "file": os.path.basename(path),
            "total_bp": total_bp,
            "valid_counted_kmers": valid_kmers,
            "distinct_kmers": len(counts),
            "count_file": count_path,
            "reused_count_file": False
        })

        print(
            f"{os.path.basename(path)}: "
            f"{total_bp:,} bp, "
            f"{valid_kmers:,} valid counted k-mers, "
            f"{len(counts):,} distinct k-mers"
        )

        # Release current sample counter before moving to next file.
        del counts

    return count_files, document_frequency, pd.DataFrame(sample_stats)


def compute_kernel_from_sorted_counts(
    count_files,
    document_frequency,
    n_samples,
    chunk_size=100000
):
    """
    Second pass:
    Merge sorted per-sample count files by k-mer and accumulate the exact
    kWIP weighted inner product kernel.

    K_ij = sum_x C_i(x) C_j(x) H(x)

    Since the files are sorted by k-mer, only one k-mer's sample counts are
    kept in memory at a time. The only dense object is the final n_samples x
    n_samples kernel matrix, which is required for pairwise distances.
    """
    K = np.zeros((n_samples, n_samples), dtype=np.float64)

    handles = []
    heap = []

    try:
        for file_idx, path in enumerate(count_files):
            handle = gzip.open(path, "rt")
            handles.append(handle)
            item = read_count_line(handle)

            if item is not None:
                kmer, count = item
                heapq.heappush(heap, (kmer, file_idx, count))

        processed_kmers = 0
        nonzero_weight_kmers = 0

        while heap:
            current_kmer = heap[0][0]
            entries = []

            while heap and heap[0][0] == current_kmer:
                _, file_idx, count = heapq.heappop(heap)
                entries.append((file_idx, count))

                next_item = read_count_line(handles[file_idx])
                if next_item is not None:
                    next_kmer, next_count = next_item
                    heapq.heappush(heap, (next_kmer, file_idx, next_count))

            df_count = document_frequency[current_kmer]
            H = binary_entropy_scalar(df_count / n_samples)

            if H > 0.0:
                nonzero_weight_kmers += 1

                for a in range(len(entries)):
                    i, ci = entries[a]
                    K[i, i] += H * ci * ci

                    for b in range(a + 1, len(entries)):
                        j, cj = entries[b]
                        value = H * ci * cj
                        K[i, j] += value
                        K[j, i] += value

            processed_kmers += 1

            if chunk_size > 0 and processed_kmers % chunk_size == 0:
                print(
                    f"  Processed {processed_kmers:,} vocabulary k-mers "
                    f"({nonzero_weight_kmers:,} nonzero-weight so far)"
                )

    finally:
        for handle in handles:
            handle.close()

    return K


def normalize_kernel_and_distance(K):
    diag = np.diag(K)
    denom = np.sqrt(np.outer(diag, diag))

    K_norm = np.zeros_like(K, dtype=np.float64)
    valid = denom > 0
    K_norm[valid] = K[valid] / denom[valid]
    K_norm = np.clip(K_norm, -1.0, 1.0)

    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * K_norm))
    np.fill_diagonal(D, 0.0)

    return K_norm, D


def save_kmer_weights(document_frequency, n_samples, output_path):
    with open(output_path, "w") as out:
        out.write("kmer,presence_frequency_F,entropy_weight_H\n")
        for kmer, df_count in sorted(document_frequency.items()):
            F = df_count / n_samples
            H = binary_entropy_scalar(F)
            out.write(f"{kmer},{F},{H}\n")


def compute_kwip_memory_safe(
    fasta_files,
    k,
    output_dir,
    canonical=True,
    chunk_size=100000,
    reuse_counts=False,
    save_weights=False
):
    """
    Memory-safe exact kWIP-style entropy-weighted inner product.

    It computes the same formula as the dense implementation, but avoids:
    - loading whole FASTA files into memory,
    - keeping all sample counters in memory together,
    - creating the huge samples x vocabulary dense count matrix.
    """
    n_samples = len(fasta_files)

    print("\n==============================")
    print(f"Computing memory-safe exact kWIP for k={k}")
    print("==============================")

    count_files, document_frequency, sample_stats_df = build_count_files(
        fasta_files=fasta_files,
        k=k,
        output_dir=output_dir,
        canonical=canonical,
        reuse_counts=reuse_counts
    )

    vocabulary_size = len(document_frequency)
    print(f"Total vocabulary size for k={k}: {vocabulary_size:,}")

    sample_stats_df.to_csv(
        os.path.join(output_dir, f"kwip_sample_count_stats_k{k}.csv"),
        index=False
    )

    print("Building exact weighted kernel from sorted count files...")
    K = compute_kernel_from_sorted_counts(
        count_files=count_files,
        document_frequency=document_frequency,
        n_samples=n_samples,
        chunk_size=chunk_size
    )

    K_norm, D = normalize_kernel_and_distance(K)

    entropy_summary = entropy_summary_from_document_frequency(
        document_frequency=document_frequency,
        n_samples=n_samples
    )

    if save_weights:
        save_kmer_weights(
            document_frequency=document_frequency,
            n_samples=n_samples,
            output_path=os.path.join(output_dir, f"kwip_kmer_entropy_weights_k{k}.csv")
        )

    return {
        "kernel_matrix": K,
        "normalized_kernel_matrix": K_norm,
        "distance_matrix": D,
        "vocabulary_size": vocabulary_size,
        "entropy_summary": entropy_summary,
        "sample_stats": sample_stats_df
    }


# ============================================================
# SILHOUETTE SCORE
# ============================================================

def compute_silhouette(distance_matrix, sample_names, group_labels):
    labels_array = np.array(group_labels)
    unique_labels = sorted(set(labels_array))

    if "unknown" in unique_labels:
        print("Silhouette score not computed because some labels are unknown.")
        global_silhouette = np.nan
        per_sample_silhouette = np.full(len(sample_names), np.nan)

    elif len(unique_labels) < 2:
        print("Silhouette score not computed because only one class exists.")
        global_silhouette = np.nan
        per_sample_silhouette = np.full(len(sample_names), np.nan)

    elif len(unique_labels) >= len(sample_names):
        print("Silhouette score not computed because every sample has its own label.")
        global_silhouette = np.nan
        per_sample_silhouette = np.full(len(sample_names), np.nan)

    else:
        global_silhouette = silhouette_score(
            distance_matrix,
            labels_array,
            metric="precomputed"
        )

        per_sample_silhouette = silhouette_samples(
            distance_matrix,
            labels_array,
            metric="precomputed"
        )

    per_sample_df = pd.DataFrame({
        "sample": sample_names,
        "label": group_labels,
        "silhouette_score": per_sample_silhouette
    })

    return global_silhouette, per_sample_df


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_heatmap(matrix, labels, title, output_path, colorbar_label):
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(label=colorbar_label)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_dendrogram(distance_matrix, labels, title, output_path):
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.ylabel("kWIP distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mds(distance_matrix, sample_names, group_labels, title, output_path):
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto"
    )
    coords = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(8, 6))
    unique_groups = sorted(set(group_labels))

    for group in unique_groups:
        idx = [i for i, g in enumerate(group_labels) if g == group]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=80, label=group)

        for i in idx:
            plt.text(coords[i, 0], coords[i, 1], sample_names[i], fontsize=8)

    plt.title(title)
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_silhouette_bar(per_sample_df, title, output_path):
    df = per_sample_df.copy()

    if df["silhouette_score"].isna().all():
        print("Silhouette plot skipped because silhouette values are NaN.")
        return

    df = df.sort_values("silhouette_score")
    x_labels = [f"{row['sample']}\n{row['label']}" for _, row in df.iterrows()]

    plt.figure(figsize=(11, 6))
    plt.bar(range(len(df)), df["silhouette_score"].values)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(df)), x_labels, rotation=90)
    plt.ylabel("Silhouette score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_entropy_distribution(H_values, title, output_path):
    if H_values.size == 0:
        print("Entropy plot skipped because there are no k-mers.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(H_values, bins=50)
    plt.xlabel("Entropy weight H(x)")
    plt.ylabel("Number of k-mers")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    fasta_dir,
    out_dir,
    k_values,
    canonical=True,
    labels_csv=None,
    label_column=None,
    sample_column=None,
    chunk_size=100000,
    reuse_counts=False,
    save_weights=False,
    no_plots=False,
    max_plot_samples=200,
    no_save_matrices=False
):
    os.makedirs(out_dir, exist_ok=True)

    fasta_files = []
    for filename in sorted(os.listdir(fasta_dir)):
        if filename.lower().endswith(FASTA_EXTENSIONS):
            fasta_files.append(os.path.join(fasta_dir, filename))

    if len(fasta_files) == 0:
        raise ValueError("No FASTA files found in input directory.")

    sample_names, group_labels, used_sample_column, used_label_column = get_group_labels(
        fasta_files=fasta_files,
        labels_csv=labels_csv,
        label_column=label_column,
        sample_column=sample_column
    )

    labels_df = pd.DataFrame({
        "sample": sample_names,
        "label": group_labels,
        "fasta_file": [os.path.basename(path) for path in fasta_files]
    })
    labels_df.to_csv(os.path.join(out_dir, "sample_labels_used.csv"), index=False)

    print("\nSamples and labels used:")
    for sample, label in zip(sample_names, group_labels):
        print(f"  {sample} -> {label}")

    print("\nLabel counts:")
    print(labels_df["label"].value_counts(dropna=False).to_string())

    all_global_results = []

    for k in k_values:
        k_out = os.path.join(out_dir, f"k{k}")
        os.makedirs(k_out, exist_ok=True)

        result = compute_kwip_memory_safe(
            fasta_files=fasta_files,
            k=k,
            output_dir=k_out,
            canonical=canonical,
            chunk_size=chunk_size,
            reuse_counts=reuse_counts,
            save_weights=save_weights
        )

        D = result["distance_matrix"]
        K = result["kernel_matrix"]
        K_norm = result["normalized_kernel_matrix"]
        entropy_summary = result["entropy_summary"]
        H_values = entropy_summary["H_values"]
        vocabulary_size = result["vocabulary_size"]

        # ----------------------------------------------------
        # Save matrices
        # ----------------------------------------------------
        if not no_save_matrices:
            pd.DataFrame(D, index=sample_names, columns=sample_names).to_csv(
                os.path.join(k_out, f"kwip_distance_matrix_k{k}.csv")
            )
            pd.DataFrame(K, index=sample_names, columns=sample_names).to_csv(
                os.path.join(k_out, f"kwip_kernel_matrix_k{k}.csv")
            )
            pd.DataFrame(K_norm, index=sample_names, columns=sample_names).to_csv(
                os.path.join(k_out, f"kwip_normalized_kernel_matrix_k{k}.csv")
            )
        else:
            print("Matrix CSV saving skipped because --no_save_matrices was used.")

        # ----------------------------------------------------
        # Entropy summary
        # ----------------------------------------------------
        weight_summary = pd.DataFrame({
            "k": [k],
            "canonical_kmers": [canonical],
            "vocabulary_size": [vocabulary_size],
            "min_entropy_weight": [entropy_summary["min_entropy_weight"]],
            "max_entropy_weight": [entropy_summary["max_entropy_weight"]],
            "mean_entropy_weight": [entropy_summary["mean_entropy_weight"]],
            "median_entropy_weight": [entropy_summary["median_entropy_weight"]],
            "zero_weight_kmers": [entropy_summary["zero_weight_kmers"]],
            "nonzero_weight_kmers": [entropy_summary["nonzero_weight_kmers"]]
        })

        weight_summary.to_csv(
            os.path.join(k_out, f"kwip_entropy_weight_summary_k{k}.csv"),
            index=False
        )

        # ----------------------------------------------------
        # Silhouette score
        # ----------------------------------------------------
        global_silhouette, per_sample_silhouette_df = compute_silhouette(
            distance_matrix=D,
            sample_names=sample_names,
            group_labels=group_labels
        )

        pd.DataFrame({
            "method": ["exact_kWIP_entropy_weighted_inner_product_memory_safe"],
            "k": [k],
            "canonical_kmers": [canonical],
            "label_source": [labels_csv if labels_csv else "filename_inference"],
            "label_column": [used_label_column if used_label_column else "filename_inference"],
            "global_silhouette_score": [global_silhouette],
            "vocabulary_size": [vocabulary_size]
        }).to_csv(
            os.path.join(k_out, f"kwip_global_silhouette_k{k}.csv"),
            index=False
        )

        per_sample_silhouette_df.to_csv(
            os.path.join(k_out, f"kwip_per_sample_silhouette_k{k}.csv"),
            index=False
        )

        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------
        skip_plots = no_plots or len(sample_names) > max_plot_samples

        if skip_plots:
            print(
                "Plots skipped "
                f"(no_plots={no_plots}, samples={len(sample_names)}, "
                f"max_plot_samples={max_plot_samples})."
            )
        else:
            plot_heatmap(
                matrix=D,
                labels=sample_names,
                title=f"kWIP distance heatmap, k={k}",
                output_path=os.path.join(k_out, f"kwip_distance_heatmap_k{k}.png"),
                colorbar_label="kWIP distance"
            )

            plot_heatmap(
                matrix=K_norm,
                labels=sample_names,
                title=f"kWIP normalized kernel similarity, k={k}",
                output_path=os.path.join(k_out, f"kwip_normalized_kernel_heatmap_k{k}.png"),
                colorbar_label="Normalized kernel similarity"
            )

            plot_dendrogram(
                distance_matrix=D,
                labels=sample_names,
                title=f"kWIP hierarchical clustering, k={k}",
                output_path=os.path.join(k_out, f"kwip_dendrogram_k{k}.png")
            )

            plot_mds(
                distance_matrix=D,
                sample_names=sample_names,
                group_labels=group_labels,
                title=f"kWIP MDS plot, k={k}",
                output_path=os.path.join(k_out, f"kwip_mds_k{k}.png")
            )

            plot_silhouette_bar(
                per_sample_df=per_sample_silhouette_df,
                title=f"kWIP per-sample silhouette scores, k={k}",
                output_path=os.path.join(k_out, f"kwip_per_sample_silhouette_k{k}.png")
            )

            plot_entropy_distribution(
                H_values=H_values,
                title=f"kWIP entropy weight distribution, k={k}",
                output_path=os.path.join(k_out, f"kwip_entropy_weight_distribution_k{k}.png")
            )

        # ----------------------------------------------------
        # Global summary
        # ----------------------------------------------------
        all_global_results.append({
            "method": "exact_kWIP_entropy_weighted_inner_product_memory_safe",
            "k": k,
            "canonical_kmers": canonical,
            "label_source": labels_csv if labels_csv else "filename_inference",
            "label_column": used_label_column if used_label_column else "filename_inference",
            "vocabulary_size": vocabulary_size,
            "global_silhouette_score": global_silhouette,
            "min_entropy_weight": entropy_summary["min_entropy_weight"],
            "max_entropy_weight": entropy_summary["max_entropy_weight"],
            "mean_entropy_weight": entropy_summary["mean_entropy_weight"],
            "median_entropy_weight": entropy_summary["median_entropy_weight"],
            "zero_weight_kmers": entropy_summary["zero_weight_kmers"],
            "nonzero_weight_kmers": entropy_summary["nonzero_weight_kmers"]
        })

        print(f"\nFinished k={k}")
        print(f"  Vocabulary size: {vocabulary_size:,}")
        print(f"  Global silhouette score: {global_silhouette}")
        print(f"  Output folder: {k_out}")

        # Drop large arrays before next k.
        del D, K, K_norm, H_values

    global_df = pd.DataFrame(all_global_results)
    global_df.to_csv(os.path.join(out_dir, "kwip_global_results.csv"), index=False)

    print("\nAll kWIP computations finished.")
    print(f"Main summary saved to: {os.path.join(out_dir, 'kwip_global_results.csv')}")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe exact kWIP-style entropy-weighted inner product "
            "from FASTA files, with labels CSV and silhouette scores."
        )
    )

    parser.add_argument(
        "--fasta_dir",
        "--input_dir",
        dest="fasta_dir",
        required=True,
        help="Directory containing FASTA files."
    )

    parser.add_argument(
        "--out_dir",
        "--output_dir",
        dest="out_dir",
        default="kwip_results",
        help="Directory where output files will be saved."
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[15, 21, 31],
        help="k values to compute. Example: --k 15 21 31"
    )

    parser.add_argument(
        "--labels_csv",
        default=None,
        help="CSV file containing sample labels. Example columns: sample,group"
    )

    parser.add_argument(
        "--label_column",
        default=None,
        help="Column in labels CSV used as class/group label. Example: --label_column group"
    )

    parser.add_argument(
        "--sample_column",
        default=None,
        help="Column in labels CSV used to match FASTA filenames. Example: --sample_column sample"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help=(
            "Number of vocabulary k-mers processed between progress messages. "
            "This script processes k-mers one group at a time, so lowering this "
            "is mainly useful for more frequent progress output."
        )
    )

    parser.add_argument(
        "--reuse_counts",
        action="store_true",
        help="Reuse existing per-sample k-mer count files if present."
    )

    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Save full per-k-mer entropy weights CSV. This can be very large."
    )

    parser.add_argument(
        "--no_save_matrices",
        action="store_true",
        help="Do not save full pairwise matrix CSV files. Useful for many samples."
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plots. Useful for large datasets."
    )

    parser.add_argument(
        "--max_plot_samples",
        type=int,
        default=200,
        help="Automatically skip plots if sample count is larger than this value."
    )

    parser.add_argument(
        "--no_canonical",
        action="store_true",
        help="Use raw k-mers instead of canonical k-mers."
    )

    args = parser.parse_args()

    run_pipeline(
        fasta_dir=args.fasta_dir,
        out_dir=args.out_dir,
        k_values=args.k,
        canonical=not args.no_canonical,
        labels_csv=args.labels_csv,
        label_column=args.label_column,
        sample_column=args.sample_column,
        chunk_size=args.chunk_size,
        reuse_counts=args.reuse_counts,
        save_weights=args.save_weights,
        no_plots=args.no_plots,
        max_plot_samples=args.max_plot_samples,
        no_save_matrices=args.no_save_matrices
    )
