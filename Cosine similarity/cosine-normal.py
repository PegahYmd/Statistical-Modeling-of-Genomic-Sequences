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

OUTPUT_DIR = "cosine_results"
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

        # Ignore k-mers containing ambiguous bases
        if "N" in kmer:
            continue

        counts[kmer] += 1

    return counts


# ============================================================
# Convert k-mer counts into shared matrix
# ============================================================

def build_kmer_matrix(fasta_files, k):
    sample_names = []
    all_counts = []
    vocabulary = set()

    for path in fasta_files:
        sample_name = os.path.basename(path).replace(".fa", "").replace(".fasta", "")
        sequence = read_fasta(path)

        counts = kmer_counts(sequence, k)

        sample_names.append(sample_name)
        all_counts.append(counts)
        vocabulary.update(counts.keys())

    vocabulary = sorted(vocabulary)

    matrix = np.zeros((len(fasta_files), len(vocabulary)), dtype=float)

    for i, counts in enumerate(all_counts):
        total = sum(counts.values())

        for j, kmer in enumerate(vocabulary):
            matrix[i, j] = counts.get(kmer, 0)

        # Convert counts to frequencies
        if total > 0:
            matrix[i, :] /= total

    return matrix, sample_names, vocabulary


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
                "Please rename files with functional/pathological or define labels manually."
            )

    return np.array(labels)


# ============================================================
# Main analysis
# ============================================================

fasta_files = sorted(glob.glob(os.path.join(FASTA_DIR, "*.fa")))

if len(fasta_files) == 0:
    raise FileNotFoundError(f"No FASTA files found in {FASTA_DIR}")

summary_results = []

for k in K_VALUES:
    print(f"\nProcessing k = {k}")

    X, sample_names, vocabulary = build_kmer_matrix(fasta_files, k)
    labels = get_labels(sample_names)

    # Cosine similarity
    sim_matrix = cosine_similarity(X)

    # Cosine distance
    dist_matrix = 1 - sim_matrix

    # Numerical safety
    np.fill_diagonal(dist_matrix, 0.0)

    # Save matrices
    sim_df = pd.DataFrame(sim_matrix, index=sample_names, columns=sample_names)
    dist_df = pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names)

    sim_df.to_csv(os.path.join(OUTPUT_DIR, f"cosine_similarity_matrix_k{k}.csv"))
    dist_df.to_csv(os.path.join(OUTPUT_DIR, f"cosine_distance_matrix_k{k}.csv"))

    # Silhouette score using precomputed distance matrix
    global_silhouette = silhouette_score(
        dist_matrix,
        labels,
        metric="precomputed"
    )

    per_sample_silhouette = silhouette_samples(
        dist_matrix,
        labels,
        metric="precomputed"
    )

    per_sample_df = pd.DataFrame({
        "sample": sample_names,
        "label": labels,
        "class": ["functional" if x == 0 else "pathological" for x in labels],
        "silhouette_score": per_sample_silhouette
    })

    per_sample_df.to_csv(
        os.path.join(OUTPUT_DIR, f"cosine_per_sample_silhouette_k{k}.csv"),
        index=False
    )

    summary_results.append({
        "k": k,
        "vocabulary_size": len(vocabulary),
        "global_silhouette_score": global_silhouette
    })

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Global Silhouette Score: {global_silhouette:.6f}")

    # ========================================================
    # Heatmap: cosine distance
    # ========================================================

    plt.figure(figsize=(9, 7))
    plt.imshow(dist_matrix, aspect="auto")
    plt.colorbar(label="Cosine distance")

    plt.xticks(range(len(sample_names)), sample_names, rotation=90)
    plt.yticks(range(len(sample_names)), sample_names)

    plt.title(f"Cosine Distance Matrix, k={k}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"cosine_distance_heatmap_k{k}.png"),
        dpi=300
    )
    plt.close()

    # ========================================================
    # Bar chart: per-sample silhouette
    # ========================================================

    plt.figure(figsize=(10, 5))
    plt.bar(sample_names, per_sample_silhouette)

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(rotation=90)
    plt.ylabel("Silhouette Score")
    plt.title(f"Per-sample Silhouette Scores Using Cosine Distance, k={k}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"cosine_per_sample_silhouette_k{k}.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# Save global summary
# ============================================================

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "cosine_global_results.csv"),
    index=False
)

print("\nDone.")
print(summary_df)