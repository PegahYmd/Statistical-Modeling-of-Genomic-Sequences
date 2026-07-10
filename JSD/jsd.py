import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score, silhouette_samples


# =========================================================
# PARAMETERS
# =========================================================

FASTA_DIR = "./1-samples"
FASTA_PATTERN = "*.fa"

K_VALUES = [15, 21, 31]

LOG_BASE = 2.0

# For JSD, smoothing is not required.
# Keep alpha = 0.0 for standard Jensen-Shannon divergence.
# If you want to test smoothed distributions, set alpha = 1.0.
ALPHA = 0.0

OUT_DIR = "JSD_results"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# FASTA READER
# =========================================================

def read_fasta(path):
    seq = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq.append(line.upper())

    return "".join(seq)


# =========================================================
# NATURAL SORTING FOR SAMPLE NAMES
# =========================================================

def natural_key(text):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", text)
    ]


# =========================================================
# LABEL EXTRACTION
# =========================================================

def infer_label(filename):
    name = filename.lower()

    if "functional" in name:
        return "functional"
    elif "pathological" in name:
        return "pathological"
    else:
        return "unknown"


# =========================================================
# K-MER COUNTING
# =========================================================

def count_kmers(seq, k):
    counts = Counter()

    valid_chars = set("ACGT")

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]

        # Skip k-mers containing N or other ambiguous characters
        if set(kmer).issubset(valid_chars):
            counts[kmer] += 1

    return counts


# =========================================================
# BUILD COUNT MATRIX
# =========================================================

def build_count_matrix(kmer_counts_list):
    vocab = sorted(set().union(*[set(c.keys()) for c in kmer_counts_list]))

    vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}

    count_matrix = np.zeros((len(kmer_counts_list), len(vocab)), dtype=np.float64)

    for i, counts in enumerate(kmer_counts_list):
        for kmer, count in counts.items():
            j = vocab_index[kmer]
            count_matrix[i, j] = count

    return count_matrix, vocab


# =========================================================
# COUNT MATRIX TO PROBABILITY MATRIX
# =========================================================

def counts_to_probabilities(count_matrix, alpha=0.0):
    n_samples, vocab_size = count_matrix.shape

    if alpha > 0:
        smoothed = count_matrix + alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        return smoothed / row_sums

    else:
        row_sums = count_matrix.sum(axis=1, keepdims=True)

        if np.any(row_sums == 0):
            raise ValueError("At least one sample has zero valid k-mers.")

        return count_matrix / row_sums


# =========================================================
# KL DIVERGENCE USED INSIDE JSD
# =========================================================

def kl_divergence(P, Q, log_base=2.0):
    """
    Computes KL(P || Q).
    Terms where P(x) = 0 contribute 0.
    """

    mask = P > 0

    return np.sum(
        P[mask] * (np.log(P[mask] / Q[mask]) / np.log(log_base))
    )


# =========================================================
# JENSEN-SHANNON DIVERGENCE
# =========================================================

def js_divergence(P, Q, log_base=2.0):
    """
    Jensen-Shannon divergence:

    JSD(P, Q) = 1/2 KL(P || M) + 1/2 KL(Q || M)

    where M = 1/2(P + Q)
    """

    M = 0.5 * (P + Q)

    return 0.5 * kl_divergence(P, M, log_base) + \
           0.5 * kl_divergence(Q, M, log_base)


# =========================================================
# PAIRWISE JSD MATRIX
# =========================================================

def compute_jsd_matrices(prob_matrix, log_base=2.0):
    n = prob_matrix.shape[0]

    jsd_matrix = np.zeros((n, n), dtype=np.float64)
    js_distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            jsd = js_divergence(
                prob_matrix[i],
                prob_matrix[j],
                log_base=log_base
            )

            js_dist = np.sqrt(jsd)

            jsd_matrix[i, j] = jsd
            jsd_matrix[j, i] = jsd

            js_distance_matrix[i, j] = js_dist
            js_distance_matrix[j, i] = js_dist

    return jsd_matrix, js_distance_matrix


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_heatmap(matrix, sample_names, title, out_path):
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrix,
        xticklabels=sample_names,
        yticklabels=sample_names,
        cmap="viridis",
        annot=True,
        fmt=".4f",
        square=True,
        cbar_kws={"label": "Distance"}
    )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_clustermap(matrix, sample_names, title, out_path):
    condensed = squareform(matrix, checks=False)

    Z = linkage(condensed, method="average")

    df = pd.DataFrame(
        matrix,
        index=sample_names,
        columns=sample_names
    )

    g = sns.clustermap(
        df,
        row_linkage=Z,
        col_linkage=Z,
        cmap="viridis",
        annot=True,
        fmt=".4f",
        figsize=(10, 10),
        cbar_kws={"label": "Distance"}
    )

    g.fig.suptitle(title, y=1.02)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# SILHOUETTE ANALYSIS
# =========================================================

def compute_silhouette(distance_matrix, labels, sample_names):
    unique_labels = sorted(set(labels))

    if "unknown" in unique_labels:
        print("Some labels are unknown. Silhouette Score will be skipped.")
        return None, None

    if len(unique_labels) < 2:
        print("Silhouette Score requires at least two classes.")
        return None, None

    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_int[label] for label in labels])

    overall_score = silhouette_score(
        distance_matrix,
        numeric_labels,
        metric="precomputed"
    )

    per_sample_scores = silhouette_samples(
        distance_matrix,
        numeric_labels,
        metric="precomputed"
    )

    result_df = pd.DataFrame({
        "sample": sample_names,
        "label": labels,
        "silhouette_score": per_sample_scores
    })

    return overall_score, result_df


# =========================================================
# MAIN ANALYSIS
# =========================================================

all_results = []

fasta_paths = sorted(
    glob.glob(os.path.join(FASTA_DIR, FASTA_PATTERN)),
    key=natural_key
)

if len(fasta_paths) == 0:
    raise FileNotFoundError(f"No FASTA files found in {FASTA_DIR}")

sample_names = [os.path.basename(p).replace(".fa", "") for p in fasta_paths]
labels = [infer_label(os.path.basename(p)) for p in fasta_paths]

print("Samples:")
for s, l in zip(sample_names, labels):
    print(f"  {s} -> {l}")

sequences = [read_fasta(path) for path in fasta_paths]


for k in K_VALUES:
    print("\n" + "="*70)
    print(f"Running Jensen-Shannon analysis for k = {k}")
    print("="*70)

    # -----------------------------------------------------
    # Count k-mers
    # -----------------------------------------------------

    kmer_counts_list = [count_kmers(seq, k) for seq in sequences]

    # -----------------------------------------------------
    # Build global vocabulary and count matrix
    # -----------------------------------------------------

    count_matrix, vocab = build_count_matrix(kmer_counts_list)

    print(f"Vocabulary size for k={k}: {len(vocab)}")

    # -----------------------------------------------------
    # Convert counts to probability distributions
    # -----------------------------------------------------

    prob_matrix = counts_to_probabilities(
        count_matrix,
        alpha=ALPHA
    )

    # -----------------------------------------------------
    # Compute JSD and JS distance matrices
    # -----------------------------------------------------

    jsd_matrix, js_distance_matrix = compute_jsd_matrices(
        prob_matrix,
        log_base=LOG_BASE
    )

    # -----------------------------------------------------
    # Save matrices
    # -----------------------------------------------------

    jsd_df = pd.DataFrame(
        jsd_matrix,
        index=sample_names,
        columns=sample_names
    )

    js_dist_df = pd.DataFrame(
        js_distance_matrix,
        index=sample_names,
        columns=sample_names
    )

    jsd_csv = os.path.join(OUT_DIR, f"JSD_matrix_k{k}.csv")
    js_dist_csv = os.path.join(OUT_DIR, f"JS_distance_matrix_k{k}.csv")

    jsd_df.to_csv(jsd_csv)
    js_dist_df.to_csv(js_dist_csv)

    # -----------------------------------------------------
    # Heatmap
    # -----------------------------------------------------

    heatmap_path = os.path.join(
        OUT_DIR,
        f"JS_distance_heatmap_k{k}.png"
    )

    plot_heatmap(
        js_distance_matrix,
        sample_names,
        title=f"Jensen-Shannon Distance Heatmap, k={k}",
        out_path=heatmap_path
    )

    # -----------------------------------------------------
    # Clustered heatmap
    # -----------------------------------------------------

    clustermap_path = os.path.join(
        OUT_DIR,
        f"JS_distance_clustermap_k{k}.png"
    )

    plot_clustermap(
        js_distance_matrix,
        sample_names,
        title=f"Jensen-Shannon Distance Clustering, k={k}",
        out_path=clustermap_path
    )

    # -----------------------------------------------------
    # Silhouette Score
    # -----------------------------------------------------

    overall_silhouette, per_sample_df = compute_silhouette(
        js_distance_matrix,
        labels,
        sample_names
    )

    if overall_silhouette is not None:
        print(f"Overall Silhouette Score for k={k}: {overall_silhouette:.6f}")

        per_sample_csv = os.path.join(
            OUT_DIR,
            f"JS_distance_per_sample_silhouette_k{k}.csv"
        )

        per_sample_df.to_csv(per_sample_csv, index=False)

    else:
        print("Silhouette Score skipped.")

    # -----------------------------------------------------
    # Store summary result
    # -----------------------------------------------------

    all_results.append({
        "k": k,
        "alpha": ALPHA,
        "log_base": LOG_BASE,
        "vocabulary_size": len(vocab),
        "overall_silhouette": overall_silhouette
    })


# =========================================================
# SAVE GLOBAL SUMMARY
# =========================================================

summary_df = pd.DataFrame(all_results)

summary_csv = os.path.join(
    OUT_DIR,
    "JS_distance_global_results.csv"
)

summary_df.to_csv(summary_csv, index=False)

print("\nDone.")
print(f"Results saved in: {OUT_DIR}")