#!/usr/bin/env python3

import os
import re
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
# FASTA READING
# ============================================================

def read_fasta(path):
    seq_parts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            seq_parts.append(line.upper())

    return "".join(seq_parts)


# ============================================================
# k-mer UTILITIES
# ============================================================

def reverse_complement(seq):
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]


def canonical_kmer(kmer):
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def count_kmers(seq, k, canonical=True):
    counts = Counter()

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]

        # Ignore k-mers containing N or other ambiguous bases
        if re.search(r"[^ACGT]", kmer):
            continue

        if canonical:
            kmer = canonical_kmer(kmer)

        counts[kmer] += 1

    return counts


# ============================================================
# LABEL INFERENCE
# ============================================================

def infer_label(filename):
    """
    Edit this function if your filenames are different.

    Current rules:
    - file contains 'functional' -> functional
    - file contains 'pathological' or 'pathogenic' -> pathological
    - sample01 to sample05 -> functional
    - sample06 to sample10 -> pathological
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


# ============================================================
# kWIP ENTROPY WEIGHTING
# ============================================================

def binary_entropy(F):
    """
    kWIP entropy weight:

    H(x) = -[F(x) log2 F(x) + (1-F(x)) log2(1-F(x))]

    F(x) is the proportion of samples where k-mer x is present.
    """

    H = np.zeros_like(F, dtype=float)

    mask = (F > 0) & (F < 1)

    H[mask] = -(
        F[mask] * np.log2(F[mask]) +
        (1 - F[mask]) * np.log2(1 - F[mask])
    )

    return H


def compute_kwip_exact(fasta_files, k, canonical=True):
    """
    Computes exact kWIP-style weighted inner product from FASTA files.

    Output:
    - count matrix
    - entropy weights
    - weighted kernel matrix
    - normalized kernel matrix
    - kWIP distance matrix
    """

    sample_names = [
        re.sub(
            r"\.(fa|fasta|fna)$",
            "",
            os.path.basename(f),
            flags=re.IGNORECASE
        )
        for f in fasta_files
    ]

    print(f"\n==============================")
    print(f"Computing exact kWIP for k={k}")
    print(f"==============================")

    sample_counters = []
    vocabulary = set()

    for path in fasta_files:
        seq = read_fasta(path)
        counts = count_kmers(seq, k, canonical=canonical)

        sample_counters.append(counts)
        vocabulary.update(counts.keys())

        print(
            f"{os.path.basename(path)}: "
            f"{len(seq):,} bp, "
            f"{len(counts):,} distinct k-mers"
        )

    vocabulary = sorted(vocabulary)
    vocab_index = {kmer: idx for idx, kmer in enumerate(vocabulary)}

    n_samples = len(fasta_files)
    n_features = len(vocabulary)

    print(f"Total vocabulary size for k={k}: {n_features:,}")

    # Count matrix: rows = samples, columns = k-mers
    X = np.zeros((n_samples, n_features), dtype=float)

    for i, counter in enumerate(sample_counters):
        for kmer, count in counter.items():
            j = vocab_index[kmer]
            X[i, j] = count

    # Presence frequency across samples
    presence = X > 0

    F = presence.sum(axis=0) / n_samples

    # Shannon entropy weights
    H = binary_entropy(F)

    # Weighted inner product:
    #
    # K_ij = sum_x C_i(x) C_j(x) H(x)
    #
    # This is equivalent to multiplying each count by sqrt(H),
    # then computing the dot product.
    X_weighted = X * np.sqrt(H)
    K = X_weighted @ X_weighted.T

    # Normalize kernel:
    #
    # K'_ij = K_ij / sqrt(K_ii K_jj)
    diag = np.diag(K)
    denom = np.sqrt(np.outer(diag, diag))

    K_norm = np.zeros_like(K, dtype=float)
    valid = denom > 0
    K_norm[valid] = K[valid] / denom[valid]

    # Numerical safety
    K_norm = np.clip(K_norm, -1.0, 1.0)

    # Convert normalized kernel to Euclidean distance:
    #
    # D_ij = sqrt(K'_ii + K'_jj - 2K'_ij)
    #
    # Since K'_ii = 1 after normalization:
    #
    # D_ij = sqrt(2 - 2K'_ij)
    D = np.sqrt(np.maximum(0, 2 - 2 * K_norm))
    np.fill_diagonal(D, 0.0)

    return {
        "sample_names": sample_names,
        "count_matrix": X,
        "vocabulary": vocabulary,
        "presence_frequency": F,
        "entropy_weights": H,
        "kernel_matrix": K,
        "normalized_kernel_matrix": K_norm,
        "distance_matrix": D
    }


# ============================================================
# SILHOUETTE SCORE
# ============================================================

def compute_silhouette(distance_matrix, sample_names, group_labels):
    """
    Computes:
    - global silhouette score
    - per-sample silhouette scores

    Because kWIP produces a distance matrix, we use:
    metric='precomputed'
    """

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
        random_state=42
    )

    coords = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(8, 6))

    unique_groups = sorted(set(group_labels))

    for group in unique_groups:
        idx = [i for i, g in enumerate(group_labels) if g == group]

        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=80,
            label=group
        )

        for i in idx:
            plt.text(
                coords[i, 0],
                coords[i, 1],
                sample_names[i],
                fontsize=8
            )

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

    x_labels = [
        f"{row['sample']}\n{row['label']}"
        for _, row in df.iterrows()
    ]

    plt.figure(figsize=(11, 6))

    plt.bar(
        range(len(df)),
        df["silhouette_score"].values
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(
        range(len(df)),
        x_labels,
        rotation=90
    )

    plt.ylabel("Silhouette score")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_entropy_distribution(H, title, output_path):
    plt.figure(figsize=(8, 6))

    plt.hist(H, bins=50)

    plt.xlabel("Entropy weight H(x)")
    plt.ylabel("Number of k-mers")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(input_dir, output_dir, k_values, canonical=True):
    os.makedirs(output_dir, exist_ok=True)

    fasta_files = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith((".fa", ".fasta", ".fna")):
            fasta_files.append(os.path.join(input_dir, filename))

    if len(fasta_files) == 0:
        raise ValueError("No FASTA files found in input directory.")

    sample_names = [
        re.sub(
            r"\.(fa|fasta|fna)$",
            "",
            os.path.basename(f),
            flags=re.IGNORECASE
        )
        for f in fasta_files
    ]

    group_labels = [
        infer_label(os.path.basename(f))
        for f in fasta_files
    ]

    print("\nSamples and inferred labels:")
    for sample, label in zip(sample_names, group_labels):
        print(f"  {sample} -> {label}")

    all_global_results = []

    for k in k_values:
        k_out = os.path.join(output_dir, f"k{k}")
        os.makedirs(k_out, exist_ok=True)

        result = compute_kwip_exact(
            fasta_files=fasta_files,
            k=k,
            canonical=canonical
        )

        D = result["distance_matrix"]
        K = result["kernel_matrix"]
        K_norm = result["normalized_kernel_matrix"]
        H = result["entropy_weights"]
        F = result["presence_frequency"]
        vocabulary = result["vocabulary"]

        # ----------------------------------------------------
        # Save matrices
        # ----------------------------------------------------

        pd.DataFrame(
            D,
            index=sample_names,
            columns=sample_names
        ).to_csv(
            os.path.join(k_out, f"kwip_distance_matrix_k{k}.csv")
        )

        pd.DataFrame(
            K,
            index=sample_names,
            columns=sample_names
        ).to_csv(
            os.path.join(k_out, f"kwip_kernel_matrix_k{k}.csv")
        )

        pd.DataFrame(
            K_norm,
            index=sample_names,
            columns=sample_names
        ).to_csv(
            os.path.join(k_out, f"kwip_normalized_kernel_matrix_k{k}.csv")
        )

        # ----------------------------------------------------
        # Save k-mer entropy weights
        # ----------------------------------------------------

        weights_df = pd.DataFrame({
            "kmer": vocabulary,
            "presence_frequency_F": F,
            "entropy_weight_H": H
        })

        weights_df.to_csv(
            os.path.join(k_out, f"kwip_kmer_entropy_weights_k{k}.csv"),
            index=False
        )

        weight_summary = pd.DataFrame({
            "k": [k],
            "canonical_kmers": [canonical],
            "vocabulary_size": [len(vocabulary)],
            "min_entropy_weight": [float(np.min(H))],
            "max_entropy_weight": [float(np.max(H))],
            "mean_entropy_weight": [float(np.mean(H))],
            "median_entropy_weight": [float(np.median(H))],
            "zero_weight_kmers": [int(np.sum(H == 0))],
            "nonzero_weight_kmers": [int(np.sum(H > 0))]
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
            "method": ["exact_kWIP_entropy_weighted_inner_product"],
            "k": [k],
            "global_silhouette_score": [global_silhouette]
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
            H=H,
            title=f"kWIP entropy weight distribution, k={k}",
            output_path=os.path.join(k_out, f"kwip_entropy_weight_distribution_k{k}.png")
        )

        # ----------------------------------------------------
        # Global summary
        # ----------------------------------------------------

        all_global_results.append({
            "method": "exact_kWIP_entropy_weighted_inner_product",
            "k": k,
            "canonical_kmers": canonical,
            "vocabulary_size": len(vocabulary),
            "global_silhouette_score": global_silhouette,
            "min_entropy_weight": float(np.min(H)),
            "max_entropy_weight": float(np.max(H)),
            "mean_entropy_weight": float(np.mean(H)),
            "median_entropy_weight": float(np.median(H)),
            "zero_weight_kmers": int(np.sum(H == 0)),
            "nonzero_weight_kmers": int(np.sum(H > 0))
        })

        print(f"\nFinished k={k}")
        print(f"  Vocabulary size: {len(vocabulary):,}")
        print(f"  Global silhouette score: {global_silhouette}")
        print(f"  Output folder: {k_out}")

    global_df = pd.DataFrame(all_global_results)

    global_df.to_csv(
        os.path.join(output_dir, "kwip_global_results.csv"),
        index=False
    )

    print("\nAll kWIP computations finished.")
    print(f"Main summary saved to: {os.path.join(output_dir, 'kwip_global_results.csv')}")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exact kWIP-style entropy-weighted inner product computation from FASTA files, including silhouette scores."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing FASTA files."
    )

    parser.add_argument(
        "--output_dir",
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
        "--no_canonical",
        action="store_true",
        help="Use raw k-mers instead of canonical k-mers."
    )

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        k_values=args.k,
        canonical=not args.no_canonical
    )