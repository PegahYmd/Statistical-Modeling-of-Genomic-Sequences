import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import glob
import os
import re
import pandas as pd
import seaborn as sns

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


# =========================================================
# PARAMETERS
# =========================================================

FASTA_PATTERN = "../1-samples/*.fa"

OUT_DIR = "SIWKL_sensitivity_results"
os.makedirs(OUT_DIR, exist_ok=True)

# k values to test
k_values = [15, 21, 31, 41, 51]

# alpha controls smoothing of sample distributions P and Q
alpha_values = [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-6]

# beta controls smoothing of global distribution G used in w(x) = -log G(x)
beta_values = [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-6]

# First, you can keep this False.
# It will test all k and alpha values using beta = 1.0.
# Later, set it to True to test all alpha-beta combinations.
RUN_FULL_BETA_GRID = True

FIXED_BETA = 1.0

# Plot best configuration at the end
PLOT_BEST = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_param_name(x):
    """
    Convert parameter values such as 1e-06 into safe filename strings.
    """
    return str(x).replace(".", "p").replace("-", "m")


def read_fasta(path):
    """
    Read a FASTA file and return the sequence as one uppercase string.
    """
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


def kmer_counts(seq, k):
    """
    Count k-mers in a sequence.
    K-mers containing N are ignored.
    """
    counts = Counter()
    L = len(seq)

    for i in range(L - k + 1):
        kmer = seq[i:i+k]

        if "N" in kmer:
            continue

        counts[kmer] += 1

    return counts


def build_vocab(count_dicts):
    """
    Build the union vocabulary of all observed k-mers.
    """
    vocab = set()

    for c in count_dicts:
        vocab.update(c.keys())

    vocab = sorted(vocab)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(vocab)}

    return vocab, kmer_to_idx


def counts_to_prob(counts, V, kmer_to_idx, alpha):
    """
    Convert k-mer counts into a smoothed probability vector.
    """
    vec = np.zeros(V, dtype=np.float64)

    for kmer, cnt in counts.items():
        vec[kmer_to_idx[kmer]] = cnt

    vec += alpha
    vec /= vec.sum()

    return vec


def build_global_distribution(count_dicts, V, kmer_to_idx, beta):
    """
    Build smoothed global k-mer distribution G(x).
    """
    global_counts = Counter()

    for c in count_dicts:
        global_counts.update(c)

    G = np.zeros(V, dtype=np.float64)

    for kmer, cnt in global_counts.items():
        G[kmer_to_idx[kmer]] = cnt

    G += beta
    G /= G.sum()

    return G


def compute_siwkl_matrix(P, W):
    """
    Compute symmetric information-weighted KL matrix.

    D_WKL(P || Q) = sum_x W(x) P(x) log(P(x) / Q(x))

    SIWKL(P,Q) = 0.5 * [D_WKL(P||Q) + D_WKL(Q||P)]
    """
    n = len(P)
    M = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            p = P[i]
            q = P[j]

            d_pq = np.sum(W * p * np.log(p / q))
            d_qp = np.sum(W * q * np.log(q / p))

            d = 0.5 * (d_pq + d_qp)

            # Safety against tiny numerical negatives
            if d < 0 and abs(d) < 1e-15:
                d = 0.0

            M[i, j] = d
            M[j, i] = d

    return M


def clean_distance_matrix(M):
    """
    Ensure the matrix is symmetric, non-negative, and has zero diagonal.
    """
    D = np.asarray(M, dtype=np.float64)

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    # sklearn silhouette requires non-negative distances
    D[D < 0] = 0.0

    return D


def intra_inter_ratio(D, class_labels):
    """
    Compute mean within-class distance, mean between-class distance,
    and their ratio.
    """
    within = []
    between = []

    n = len(class_labels)

    for i in range(n):
        for j in range(i + 1, n):
            if class_labels[i] == class_labels[j]:
                within.append(D[i, j])
            else:
                between.append(D[i, j])

    mean_within = np.mean(within)
    mean_between = np.mean(between)
    ratio = mean_within / mean_between

    return mean_within, mean_between, ratio


def dunn_index(D, class_labels):
    """
    Dunn index for distance matrix.

    Dunn = minimum inter-class distance / maximum intra-class distance

    Higher is better.
    """
    intra = []
    inter = []

    n = len(class_labels)

    for i in range(n):
        for j in range(i + 1, n):
            if class_labels[i] == class_labels[j]:
                intra.append(D[i, j])
            else:
                inter.append(D[i, j])

    max_intra = np.max(intra)
    min_inter = np.min(inter)

    if max_intra == 0:
        return np.nan

    return min_inter / max_intra


def nearest_neighbor_accuracy(D, class_labels):
    """
    Leave-one-out nearest-neighbor accuracy.
    """
    n = len(class_labels)
    correct = 0

    for i in range(n):
        distances = D[i].copy()
        distances[i] = np.inf

        nearest = np.argmin(distances)

        if class_labels[i] == class_labels[nearest]:
            correct += 1

    return correct / n


def plot_heatmap(D, labels, title):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        D,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=False
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_clustermap(D, labels, title):
    D = clean_distance_matrix(D)

    condensed = squareform(D)
    Z = linkage(condensed, method="average")

    g = sns.clustermap(
        D,
        row_linkage=Z,
        col_linkage=Z,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        figsize=(8, 8)
    )

    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.9)
    plt.show()


def plot_mds(D, labels, title):
    D = clean_distance_matrix(D)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42
    )

    coords = mds.fit_transform(D)

    plt.figure(figsize=(8, 6))

    for i, label in enumerate(labels):
        plt.scatter(coords[i, 0], coords[i, 1])
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=8)

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# LOAD FASTA FILES
# =========================================================

files = sorted(glob.glob(FASTA_PATTERN))

if len(files) == 0:
    raise RuntimeError(f"No FASTA files found with pattern: {FASTA_PATTERN}")

print("Found FASTA files:")
for f in files:
    print(" -", f)

sample_labels = [os.path.basename(f).replace(".fa", "") for f in files]

# Class labels:
# functional = 0
# pathological = 1
class_labels = []

for f in files:
    name = os.path.basename(f).lower()

    if "functional" in name:
        class_labels.append(0)
    elif "pathological" in name:
        class_labels.append(1)
    else:
        raise RuntimeError(f"Cannot detect class from filename: {f}")

class_labels = np.array(class_labels)

print("\nSample labels:")
for label, cls in zip(sample_labels, class_labels):
    class_name = "functional" if cls == 0 else "pathological"
    print(f"{label}: {class_name}")

print("\nReading FASTA sequences...")
sequences = [read_fasta(f) for f in files]

print("\nSequence lengths:")
for label, seq in zip(sample_labels, sequences):
    print(f"{label}: {len(seq)} bp")


# =========================================================
# MAIN EXPERIMENT
# =========================================================

all_results = []
all_sample_results = []

best_score = -999
best_config = None
best_matrix = None

for k in k_values:

    print("\n==================================================")
    print(f"Processing k = {k}")
    print("==================================================")

    # -------------------------------
    # Count k-mers
    # -------------------------------
    print("Counting k-mers...")

    count_dicts = [kmer_counts(seq, k) for seq in sequences]

    real_kmer_counts = np.array([sum(c.values()) for c in count_dicts])
    N_ref = np.mean(real_kmer_counts)

    print("Real k-mer counts per sample:")
    for label, n_kmers in zip(sample_labels, real_kmer_counts):
        print(f"{label}: {n_kmers}")

    # -------------------------------
    # Build vocabulary
    # -------------------------------
    print("Building vocabulary...")

    vocab, kmer_to_idx = build_vocab(count_dicts)
    V = len(vocab)

    if V == 0:
        raise RuntimeError(f"Vocabulary is empty for k={k}")

    print(f"Vocabulary size V for k={k}: {V}")
    print(f"Average real k-mer count N: {N_ref:.2f}")

    # -------------------------------
    # Print smoothing mass information
    # -------------------------------
    print("\nAlpha pseudo-count mass check:")
    for alpha in alpha_values:
        pseudo_mass = alpha * V
        ratio = pseudo_mass / N_ref

        print(
            f"alpha={alpha:<8} "
            f"pseudo_mass={pseudo_mass:.4f} "
            f"pseudo/real ratio={ratio:.6f}"
        )

    # -------------------------------
    # Decide beta values for this run
    # -------------------------------
    if RUN_FULL_BETA_GRID:
        beta_loop_values = beta_values
    else:
        beta_loop_values = [FIXED_BETA]

    # -------------------------------
    # Loop over alpha and beta
    # -------------------------------
    for alpha in alpha_values:

        print("\n------------------------------------------")
        print(f"Building sample probability vectors for k={k}, alpha={alpha}")
        print("------------------------------------------")

        P = [
            counts_to_prob(c, V, kmer_to_idx, alpha)
            for c in count_dicts
        ]

        P = np.asarray(P, dtype=np.float64)

        for beta in beta_loop_values:

            print("\nRunning configuration:")
            print(f"k={k}, alpha={alpha}, beta={beta}")

            # -------------------------------
            # Build global distribution G
            # -------------------------------
            G = build_global_distribution(count_dicts, V, kmer_to_idx, beta)

            # -------------------------------
            # Weight function
            # w(x) = -log G(x)
            # -------------------------------
            W = -np.log(G)

            # Normalize weights for numerical stability
            W = W / np.max(W)

            # -------------------------------
            # Compute SIWKL matrix
            # -------------------------------
            M = compute_siwkl_matrix(P, W)
            D = clean_distance_matrix(M)

            # -------------------------------
            # Save distance matrix
            # -------------------------------
            matrix_filename = (
                f"SIWKL_matrix_"
                f"k{k}_"
                f"alpha{safe_param_name(alpha)}_"
                f"beta{safe_param_name(beta)}.txt"
            )

            matrix_path = os.path.join(OUT_DIR, matrix_filename)
            np.savetxt(matrix_path, D, fmt="%.6e")

            # -------------------------------
            # Silhouette score
            # -------------------------------
            sil_global = silhouette_score(
                D,
                class_labels,
                metric="precomputed"
            )

            sil_per_sample = silhouette_samples(
                D,
                class_labels,
                metric="precomputed"
            )

            mean_functional_sil = np.mean(sil_per_sample[class_labels == 0])
            mean_pathological_sil = np.mean(sil_per_sample[class_labels == 1])

            # -------------------------------
            # Other separation measures
            # -------------------------------
            mean_within, mean_between, within_between_ratio = intra_inter_ratio(
                D,
                class_labels
            )

            dunn = dunn_index(D, class_labels)

            nn_acc = nearest_neighbor_accuracy(D, class_labels)

            # -------------------------------
            # Store global result
            # -------------------------------
            result = {
                "method": "SIWKL",
                "k": k,
                "alpha": alpha,
                "beta": beta,
                "vocab_size": V,
                "avg_real_kmers": N_ref,
                "alpha_pseudo_mass": alpha * V,
                "alpha_pseudo_real_ratio": (alpha * V) / N_ref,
                "silhouette_score": sil_global,
                "mean_functional_silhouette": mean_functional_sil,
                "mean_pathological_silhouette": mean_pathological_sil,
                "mean_within_class_distance": mean_within,
                "mean_between_class_distance": mean_between,
                "within_between_ratio": within_between_ratio,
                "dunn_index": dunn,
                "nearest_neighbor_accuracy": nn_acc,
                "matrix_file": matrix_filename
            }

            all_results.append(result)

            # -------------------------------
            # Store per-sample results
            # -------------------------------
            for label, cls, s_value in zip(sample_labels, class_labels, sil_per_sample):
                class_name = "functional" if cls == 0 else "pathological"

                all_sample_results.append({
                    "method": "SIWKL",
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "sample": label,
                    "class": class_name,
                    "silhouette_value": s_value
                })

            # -------------------------------
            # Print summary
            # -------------------------------
            print(f"Silhouette Score: {sil_global:.6f}")
            print(f"Mean functional silhouette: {mean_functional_sil:.6f}")
            print(f"Mean pathological silhouette: {mean_pathological_sil:.6f}")
            print(f"Within/between ratio: {within_between_ratio:.6f}")
            print(f"Dunn index: {dunn:.6f}")
            print(f"Nearest-neighbor accuracy: {nn_acc:.6f}")
            print(f"Saved matrix: {matrix_path}")

            # -------------------------------
            # Track best result
            # -------------------------------
            if sil_global > best_score:
                best_score = sil_global
                best_config = result
                best_matrix = D.copy()


# =========================================================
# SAVE RESULTS TABLES
# =========================================================

results_df = pd.DataFrame(all_results)
sample_results_df = pd.DataFrame(all_sample_results)

results_csv = os.path.join(OUT_DIR, "SIWKL_global_results.csv")
sample_csv = os.path.join(OUT_DIR, "SIWKL_per_sample_silhouette.csv")

results_df.to_csv(results_csv, index=False)
sample_results_df.to_csv(sample_csv, index=False)

print("\n==================================================")
print("FINAL RESULTS TABLE")
print("==================================================")
print(results_df)

print(f"\nSaved global results table to: {results_csv}")
print(f"Saved per-sample silhouette table to: {sample_csv}")


# =========================================================
# PRINT BEST CONFIGURATION
# =========================================================

print("\n==================================================")
print("BEST CONFIGURATION BASED ON SILHOUETTE SCORE")
print("==================================================")

for key, value in best_config.items():
    print(f"{key}: {value}")


# =========================================================
# PLOT BEST CONFIGURATION
# =========================================================

if PLOT_BEST and best_matrix is not None:

    title_suffix = (
        f"k={best_config['k']}, "
        f"alpha={best_config['alpha']}, "
        f"beta={best_config['beta']}, "
        f"sil={best_config['silhouette_score']:.4f}"
    )

    plot_heatmap(
        best_matrix,
        sample_labels,
        f"Best SIWKL Distance Matrix\n{title_suffix}"
    )

    plot_clustermap(
        best_matrix,
        sample_labels,
        f"Best SIWKL Clustered Heatmap\n{title_suffix}"
    )

    plot_mds(
        best_matrix,
        sample_labels,
        f"Best SIWKL MDS Projection\n{title_suffix}"
    )