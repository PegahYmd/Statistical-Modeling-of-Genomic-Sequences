import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples


# ============================================================
# Settings
# ============================================================

FASTA_DIR = "./1-samples"
K_VALUES = [15, 21, 31]

OUTPUT_DIR = "cosine_length_weighted_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FASTA reader
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
# k-mer counting
# ============================================================

def kmer_counts(sequence, k):
    counts = Counter()

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]

        # Ignore ambiguous k-mers
        if "N" in kmer:
            continue

        counts[kmer] += 1

    return counts


# ============================================================
# Build k-mer frequency matrix
# ============================================================

def build_kmer_matrix(fasta_files, k):
    sample_names = []
    sequence_lengths = []
    all_counts = []
    vocabulary = set()

    for path in fasta_files:
        sample_name = os.path.basename(path)
        sample_name = sample_name.replace(".fasta", "").replace(".fa", "")

        sequence = read_fasta(path)
        sequence_lengths.append(len(sequence))

        counts = kmer_counts(sequence, k)

        sample_names.append(sample_name)
        all_counts.append(counts)
        vocabulary.update(counts.keys())

    vocabulary = sorted(vocabulary)

    X = np.zeros((len(fasta_files), len(vocabulary)), dtype=float)

    for i, counts in enumerate(all_counts):
        total = sum(counts.values())

        for j, kmer in enumerate(vocabulary):
            X[i, j] = counts.get(kmer, 0)

        # Convert counts to frequencies
        if total > 0:
            X[i, :] /= total

    return X, sample_names, np.array(sequence_lengths), vocabulary


# ============================================================
# Label extraction
# ============================================================

def get_labels(sample_names):
    labels = []

    for name in sample_names:
        lower = name.lower()

        if "functional" in lower:
            labels.append(0)

        elif "pathological" in lower:
            labels.append(1)

        else:
            raise ValueError(
                f"Cannot infer label for sample '{name}'. "
                "Please rename files with functional/pathological "
                "or define labels manually."
            )

    return np.array(labels)


# ============================================================
# Length correction matrix
# ============================================================

def compute_length_weight_matrix(sequence_lengths):
    """
    Computes a pairwise length-weight matrix.

    The factor is:

        factor(i,j) = max(0, 1 - |Li - Lj| / ((Li + Lj)/2))

    If two sequences have the same length, factor = 1.
    If they are very different in length, factor becomes smaller.

    This is a bounded version of the length penalty idea.
    """

    n = len(sequence_lengths)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            Li = sequence_lengths[i]
            Lj = sequence_lengths[j]

            mean_len = (Li + Lj) / 2

            if mean_len == 0:
                W[i, j] = 0.0
            else:
                penalty = abs(Li - Lj) / mean_len
                W[i, j] = max(0.0, 1.0 - penalty)

    return W


# ============================================================
# Save heatmap
# ============================================================

def save_heatmap(matrix, sample_names, title, colorbar_label, output_path):
    plt.figure(figsize=(9, 7))

    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=colorbar_label)

    plt.xticks(range(len(sample_names)), sample_names, rotation=90)
    plt.yticks(range(len(sample_names)), sample_names)

    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Save per-sample silhouette bar chart
# ============================================================

def save_silhouette_barplot(sample_names, values, title, output_path):
    plt.figure(figsize=(10, 5))

    plt.bar(sample_names, values)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(rotation=90)
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main analysis
# ============================================================

fasta_files = sorted(glob.glob(os.path.join(FASTA_DIR, "*.fa")))

if len(fasta_files) == 0:
    raise FileNotFoundError(f"No FASTA files found in {FASTA_DIR}")

all_global_results = []

for k in K_VALUES:
    print("=" * 70)
    print(f"Processing k = {k}")
    print("=" * 70)

    # --------------------------------------------------------
    # Build k-mer matrix
    # --------------------------------------------------------

    X, sample_names, sequence_lengths, vocabulary = build_kmer_matrix(
        fasta_files,
        k
    )

    labels = get_labels(sample_names)

    # --------------------------------------------------------
    # Save sequence length information
    # --------------------------------------------------------

    length_df = pd.DataFrame({
        "sample": sample_names,
        "class": ["functional" if x == 0 else "pathological" for x in labels],
        "sequence_length": sequence_lengths
    })

    length_df.to_csv(
        os.path.join(OUTPUT_DIR, f"sequence_lengths_k{k}.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Raw cosine similarity and distance
    # --------------------------------------------------------

    raw_similarity = cosine_similarity(X)
    raw_distance = 1.0 - raw_similarity

    np.fill_diagonal(raw_distance, 0.0)

    # --------------------------------------------------------
    # Length-weighted cosine similarity and distance
    # --------------------------------------------------------

    length_weight_matrix = compute_length_weight_matrix(sequence_lengths)

    weighted_similarity = raw_similarity * length_weight_matrix
    weighted_distance = 1.0 - weighted_similarity

    np.fill_diagonal(weighted_distance, 0.0)

    # --------------------------------------------------------
    # Save matrices
    # --------------------------------------------------------

    pd.DataFrame(
        raw_similarity,
        index=sample_names,
        columns=sample_names
    ).to_csv(
        os.path.join(OUTPUT_DIR, f"raw_cosine_similarity_matrix_k{k}.csv")
    )

    pd.DataFrame(
        raw_distance,
        index=sample_names,
        columns=sample_names
    ).to_csv(
        os.path.join(OUTPUT_DIR, f"raw_cosine_distance_matrix_k{k}.csv")
    )

    pd.DataFrame(
        length_weight_matrix,
        index=sample_names,
        columns=sample_names
    ).to_csv(
        os.path.join(OUTPUT_DIR, f"length_weight_matrix_k{k}.csv")
    )

    pd.DataFrame(
        weighted_similarity,
        index=sample_names,
        columns=sample_names
    ).to_csv(
        os.path.join(OUTPUT_DIR, f"length_weighted_cosine_similarity_matrix_k{k}.csv")
    )

    pd.DataFrame(
        weighted_distance,
        index=sample_names,
        columns=sample_names
    ).to_csv(
        os.path.join(OUTPUT_DIR, f"length_weighted_cosine_distance_matrix_k{k}.csv")
    )

    # --------------------------------------------------------
    # Silhouette scores
    # --------------------------------------------------------

    raw_global_silhouette = silhouette_score(
        raw_distance,
        labels,
        metric="precomputed"
    )

    weighted_global_silhouette = silhouette_score(
        weighted_distance,
        labels,
        metric="precomputed"
    )

    raw_per_sample_silhouette = silhouette_samples(
        raw_distance,
        labels,
        metric="precomputed"
    )

    weighted_per_sample_silhouette = silhouette_samples(
        weighted_distance,
        labels,
        metric="precomputed"
    )

    per_sample_df = pd.DataFrame({
        "sample": sample_names,
        "label": labels,
        "class": ["functional" if x == 0 else "pathological" for x in labels],
        "raw_cosine_silhouette": raw_per_sample_silhouette,
        "length_weighted_cosine_silhouette": weighted_per_sample_silhouette
    })

    per_sample_df.to_csv(
        os.path.join(OUTPUT_DIR, f"cosine_per_sample_silhouette_k{k}.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Save global results
    # --------------------------------------------------------

    all_global_results.append({
        "k": k,
        "vocabulary_size": len(vocabulary),
        "min_sequence_length": int(np.min(sequence_lengths)),
        "max_sequence_length": int(np.max(sequence_lengths)),
        "mean_sequence_length": float(np.mean(sequence_lengths)),
        "raw_cosine_global_silhouette": raw_global_silhouette,
        "length_weighted_cosine_global_silhouette": weighted_global_silhouette
    })

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_heatmap(
        raw_distance,
        sample_names,
        f"Raw Cosine Distance Matrix, k={k}",
        "Raw cosine distance",
        os.path.join(OUTPUT_DIR, f"raw_cosine_distance_heatmap_k{k}.png")
    )

    save_heatmap(
        weighted_distance,
        sample_names,
        f"Length-Weighted Cosine Distance Matrix, k={k}",
        "Length-weighted cosine distance",
        os.path.join(OUTPUT_DIR, f"length_weighted_cosine_distance_heatmap_k{k}.png")
    )

    save_heatmap(
        length_weight_matrix,
        sample_names,
        f"Length Weight Matrix, k={k}",
        "Length weight",
        os.path.join(OUTPUT_DIR, f"length_weight_matrix_heatmap_k{k}.png")
    )

    save_silhouette_barplot(
        sample_names,
        raw_per_sample_silhouette,
        f"Per-Sample Silhouette Scores, Raw Cosine, k={k}",
        os.path.join(OUTPUT_DIR, f"raw_cosine_per_sample_silhouette_k{k}.png")
    )

    save_silhouette_barplot(
        sample_names,
        weighted_per_sample_silhouette,
        f"Per-Sample Silhouette Scores, Length-Weighted Cosine, k={k}",
        os.path.join(OUTPUT_DIR, f"length_weighted_cosine_per_sample_silhouette_k{k}.png")
    )

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Min sequence length: {np.min(sequence_lengths)}")
    print(f"Max sequence length: {np.max(sequence_lengths)}")
    print(f"Raw cosine silhouette: {raw_global_silhouette:.6f}")
    print(f"Length-weighted cosine silhouette: {weighted_global_silhouette:.6f}")


# ============================================================
# Save final global summary
# ============================================================

global_results_df = pd.DataFrame(all_global_results)

global_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "cosine_global_results_comparison.csv"),
    index=False
)

print("\nDone.")
print(global_results_df)