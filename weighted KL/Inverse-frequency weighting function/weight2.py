import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import Counter

from sklearn.metrics import silhouette_score, silhouette_samples


# =========================================================
# PARAMETERS
# =========================================================

k_values = [15, 21, 31]

alpha_values = [0.001, 0.01, 0.1, 1.0]

# beta=True means we use beta smoothing in global G(x)
# G(x) = (C(x) + beta) / (sum C(x) + beta * V)
beta_values = [0.001, 0.01, 0.1, 1.0]

fasta_pattern = "../1-samples/*.fa"

method_name = "SIWKL_inverse_G_global_beta_true_uncapped"

# This is the important setting:
# uncapped means do NOT limit max weight.
cap_weights = False
weight_cap_value = None


# =========================================================
# FASTA READER
# =========================================================

def read_fasta(path):
    """
    Reads a FASTA file and returns the full sequence as one uppercase string.
    Header lines starting with '>' are ignored.
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


# =========================================================
# K-MER COUNTER
# =========================================================

def count_kmers(sequence, k):
    """
    Counts valid DNA k-mers in a sequence.
    Only k-mers containing A, C, G, T are counted.
    """
    counts = Counter()
    valid_chars = set("ACGT")

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]

        if set(kmer).issubset(valid_chars):
            counts[kmer] += 1

    return counts


# =========================================================
# WEIGHTED KL FUNCTION
# =========================================================

def weighted_kl(P, Q, w):
    """
    Weighted KL-like score:

    D_w(P || Q) = sum_x w(x) P(x) log(P(x) / Q(x))

    Here:
    w(x) = 1 / G(x)

    Important:
    This is a weighted KL-based score, not a formal metric.
    """
    return np.sum(w * P * np.log(P / Q))


# =========================================================
# SAMPLE STATUS BASED ON PER-SAMPLE SILHOUETTE
# =========================================================

def classify_sample_status(score):
    """
    Interprets per-sample silhouette scores.
    """
    if score > 0.10:
        return "clear"
    elif score >= 0:
        return "weak_ambiguous"
    else:
        return "possibly_misplaced"


# =========================================================
# LOAD FASTA FILES
# =========================================================

files = sorted(glob.glob(fasta_pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No FASTA files found using pattern: {fasta_pattern}")

sample_names = [os.path.basename(f) for f in files]
N = len(files)

print("Loaded FASTA files:")
for name in sample_names:
    print("  ", name)

print(f"\nNumber of samples: {N}")


# =========================================================
# READ SEQUENCES ONCE
# =========================================================

sequences = [read_fasta(f) for f in files]

print("\nFinished reading FASTA files.")


# =========================================================
# CREATE LABELS FROM FILE NAMES
# =========================================================
# This assumes filenames contain:
# "functional" or "pathological"

labels = []

for name in sample_names:
    lower = name.lower()

    if "functional" in lower:
        labels.append(0)
    elif "pathological" in lower:
        labels.append(1)
    else:
        labels.append(-1)

labels = np.array(labels)

if not np.all(labels != -1):
    raise ValueError(
        "Some labels were not detected. "
        "Make sure filenames contain 'functional' or 'pathological'."
    )

if len(set(labels)) < 2:
    raise ValueError("Silhouette score requires at least two classes.")

label_names = ["functional" if lab == 0 else "pathological" for lab in labels]

print("\nDetected labels:")
for name, label in zip(sample_names, label_names):
    print(f"  {name:35s}  {label}")


# =========================================================
# RESULTS STORAGE
# =========================================================

overall_results = []
all_per_sample_results = []


# =========================================================
# MAIN LOOP OVER k
# =========================================================

for k in k_values:

    print("\n" + "=" * 100)
    print(f"Preparing k-mer count matrix for k = {k}")
    print("=" * 100)

    # =====================================================
    # COUNT K-MERS
    # =====================================================

    all_counts = []

    for sequence in sequences:
        counts = count_kmers(sequence, k)
        all_counts.append(counts)

    print("Finished counting k-mers.")

    # =====================================================
    # BUILD GLOBAL VOCABULARY
    # =====================================================
    # V = number of unique k-mers across all FASTA files.

    vocab = sorted(set().union(*[set(c.keys()) for c in all_counts]))
    vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}

    V = len(vocab)

    print(f"Vocabulary size for k={k}: {V}")

    # =====================================================
    # BUILD COUNT MATRIX
    # =====================================================
    # count_matrix[i, j] = count of k-mer j in sample i

    count_matrix = np.zeros((N, V), dtype=np.float64)

    for i, counts in enumerate(all_counts):
        for kmer, count in counts.items():
            j = vocab_index[kmer]
            count_matrix[i, j] = count

    row_sums = count_matrix.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0):
        raise ValueError(f"At least one FASTA file produced zero valid k-mers for k={k}.")

    # =====================================================
    # ZERO ANALYSIS BEFORE SMOOTHING
    # =====================================================

    zero_entries = np.sum(count_matrix == 0)
    total_entries = count_matrix.size
    zero_percentage = 100.0 * zero_entries / total_entries

    print(f"Zero entries before smoothing: {zero_entries}")
    print(f"Zero percentage before smoothing: {zero_percentage:.4f}%")

    # =====================================================
    # GLOBAL K-MER COUNTS
    # =====================================================

    global_counts = count_matrix.sum(axis=0)
    global_total = global_counts.sum()

    if np.any(global_counts == 0):
        raise ValueError("Unexpected zero global count found in vocabulary.")

    # =====================================================
    # LOOP OVER alpha AND beta VALUES
    # =====================================================

    for alpha in alpha_values:

        # -------------------------------------------------
        # Sample-level smoothed probability distribution
        # P_i(x) = (c_i(x) + alpha) / (sum_y c_i(y) + alpha V)
        # -------------------------------------------------

        P = (count_matrix + alpha) / (row_sums + alpha * V)

        zero_probs_after_alpha = np.sum(P == 0)

        if zero_probs_after_alpha != 0:
            raise ValueError("Zero probabilities remain after alpha smoothing.")

        for beta in beta_values:

            print("\n" + "-" * 90)
            print(
                f"Running {method_name}, "
                f"k={k}, alpha={alpha}, beta={beta}, uncapped=True"
            )
            print("-" * 90)

            # =================================================
            # GLOBAL DISTRIBUTION WITH beta=True
            # =================================================
            # G(x) = (C(x) + beta) / (sum_y C(y) + beta V)

            G = (global_counts + beta) / (global_total + beta * V)

            # Safety checks
            if np.any(G <= 0):
                raise ValueError("G(x) contains zero or negative values.")

            # =================================================
            # UNCAPPED INVERSE GLOBAL FREQUENCY WEIGHT
            # =================================================
            # w(x) = 1 / G(x)
            #
            # uncapped means:
            # do NOT apply:
            # w = min(w, weight_cap_value)

            w = 1.0 / G

            if cap_weights:
                w = np.minimum(w, weight_cap_value)

            weight_min = np.min(w)
            weight_max = np.max(w)
            weight_mean = np.mean(w)

            print("Weight statistics:")
            print(f"  min weight:  {weight_min:.6e}")
            print(f"  max weight:  {weight_max:.6e}")
            print(f"  mean weight: {weight_mean:.6e}")

            # =================================================
            # DIRECTED WEIGHTED KL MATRIX
            # =================================================

            D_directed = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                for j in range(N):
                    if i != j:
                        D_directed[i, j] = weighted_kl(P[i], P[j], w)

            # =================================================
            # SYMMETRIC WEIGHTED KL MATRIX
            # =================================================
            # Used for silhouette score:
            #
            # D_sym(P_i, P_j) =
            # 0.5 * [D_w(P_i || P_j) + D_w(P_j || P_i)]

            D_symmetric = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                for j in range(N):
                    D_symmetric[i, j] = 0.5 * (
                        D_directed[i, j] + D_directed[j, i]
                    )

            np.fill_diagonal(D_symmetric, 0)

            # =================================================
            # CHECK NEGATIVE DISTANCES
            # =================================================
            # Weighted KL-like scores are not guaranteed to be
            # valid metrics. If negative values appear, we report
            # them and clip them to zero only for silhouette_score.

            negative_entries = np.sum(D_symmetric < 0)
            min_distance = np.min(D_symmetric)
            max_distance = np.max(D_symmetric)

            print("Distance statistics before cleanup:")
            print(f"  min symmetric score: {min_distance:.6e}")
            print(f"  max symmetric score: {max_distance:.6e}")
            print(f"  negative entries:    {negative_entries}")

            D_for_silhouette = D_symmetric.copy()

            if negative_entries > 0:
                print(
                    "WARNING: Negative entries found in the symmetric weighted KL matrix. "
                    "They will be clipped to 0 only for silhouette computation."
                )
                D_for_silhouette[D_for_silhouette < 0] = 0

            np.fill_diagonal(D_for_silhouette, 0)

            # =================================================
            # OVERALL SILHOUETTE SCORE
            # =================================================

            overall_sil = silhouette_score(
                D_for_silhouette,
                labels,
                metric="precomputed"
            )

            print(f"Overall Silhouette Score: {overall_sil:.4f}")

            # =================================================
            # PER-SAMPLE SILHOUETTE SCORES
            # =================================================

            sample_silhouettes = silhouette_samples(
                D_for_silhouette,
                labels,
                metric="precomputed"
            )

            print("\nPer-sample Silhouette Scores:")
            print("-" * 105)
            print(f"{'Sample':35s} {'Class':15s} {'Score':>12s} {'Status':>22s}")
            print("-" * 105)

            for name, class_name, sample_score in zip(
                sample_names,
                label_names,
                sample_silhouettes
            ):
                status = classify_sample_status(sample_score)

                print(
                    f"{name:35s} "
                    f"{class_name:15s} "
                    f"{sample_score:12.4f} "
                    f"{status:22s}"
                )

                all_per_sample_results.append({
                    "method": method_name,
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "uncapped": True,
                    "sample": name,
                    "class": class_name,
                    "sample_silhouette_score": sample_score,
                    "status": status
                })

            # =================================================
            # SAVE OVERALL RESULT
            # =================================================

            overall_results.append({
                "method": method_name,
                "k": k,
                "alpha": alpha,
                "beta": beta,
                "uncapped": True,
                "vocabulary_size": V,
                "zero_percentage_before_smoothing": zero_percentage,
                "zero_probs_after_alpha_smoothing": zero_probs_after_alpha,
                "weight_min": weight_min,
                "weight_max": weight_max,
                "weight_mean": weight_mean,
                "negative_entries_before_cleanup": negative_entries,
                "min_symmetric_score_before_cleanup": min_distance,
                "max_symmetric_score_before_cleanup": max_distance,
                "overall_silhouette_score": overall_sil
            })


# =========================================================
# PRINT OVERALL RESULTS TABLE
# =========================================================

print("\n" + "=" * 130)
print("FULL RESULTS TABLE: SIWKL inverse G, beta=True, uncapped")
print("=" * 130)

print(
    f"{'Method':<40} "
    f"{'k':>5} "
    f"{'alpha':>10} "
    f"{'beta':>10} "
    f"{'Vocab':>12} "
    f"{'w_mean':>14} "
    f"{'w_max':>14} "
    f"{'Neg':>8} "
    f"{'Silhouette':>14}"
)

print("-" * 130)

for r in overall_results:
    print(
        f"{r['method']:<40} "
        f"{r['k']:>5} "
        f"{r['alpha']:>10.3g} "
        f"{r['beta']:>10.3g} "
        f"{r['vocabulary_size']:>12} "
        f"{r['weight_mean']:>14.4e} "
        f"{r['weight_max']:>14.4e} "
        f"{r['negative_entries_before_cleanup']:>8} "
        f"{r['overall_silhouette_score']:>14.4f}"
    )


# =========================================================
# SAVE OVERALL RESULTS CSV
# =========================================================

overall_file = "SIWKL_inverse_G_global_beta_true_uncapped_results.csv"

with open(overall_file, "w") as f:
    f.write(
        "method,k,alpha,beta,uncapped,vocabulary_size,"
        "zero_percentage_before_smoothing,"
        "zero_probs_after_alpha_smoothing,"
        "weight_min,weight_max,weight_mean,"
        "negative_entries_before_cleanup,"
        "min_symmetric_score_before_cleanup,"
        "max_symmetric_score_before_cleanup,"
        "overall_silhouette_score\n"
    )

    for r in overall_results:
        f.write(
            f"{r['method']},"
            f"{r['k']},"
            f"{r['alpha']},"
            f"{r['beta']},"
            f"{r['uncapped']},"
            f"{r['vocabulary_size']},"
            f"{r['zero_percentage_before_smoothing']:.8f},"
            f"{r['zero_probs_after_alpha_smoothing']},"
            f"{r['weight_min']:.12e},"
            f"{r['weight_max']:.12e},"
            f"{r['weight_mean']:.12e},"
            f"{r['negative_entries_before_cleanup']},"
            f"{r['min_symmetric_score_before_cleanup']:.12e},"
            f"{r['max_symmetric_score_before_cleanup']:.12e},"
            f"{r['overall_silhouette_score']:.8f}\n"
        )

print(f"\nSaved overall results CSV: {overall_file}")


# =========================================================
# SAVE PER-SAMPLE RESULTS CSV
# =========================================================

per_sample_file = "SIWKL_inverse_G_global_beta_true_uncapped_per_sample_silhouette.csv"

with open(per_sample_file, "w") as f:
    f.write(
        "method,k,alpha,beta,uncapped,sample,class,"
        "sample_silhouette_score,status\n"
    )

    for r in all_per_sample_results:
        f.write(
            f"{r['method']},"
            f"{r['k']},"
            f"{r['alpha']},"
            f"{r['beta']},"
            f"{r['uncapped']},"
            f"{r['sample']},"
            f"{r['class']},"
            f"{r['sample_silhouette_score']:.8f},"
            f"{r['status']}\n"
        )

print(f"Saved per-sample results CSV: {per_sample_file}")


# =========================================================
# FIND BEST SETTING
# =========================================================

best_result = max(
    overall_results,
    key=lambda r: r["overall_silhouette_score"]
)

print("\n" + "=" * 90)
print("BEST SETTING")
print("=" * 90)

print(f"Method: {best_result['method']}")
print(f"k: {best_result['k']}")
print(f"alpha: {best_result['alpha']}")
print(f"beta: {best_result['beta']}")
print(f"uncapped: {best_result['uncapped']}")
print(f"vocabulary size: {best_result['vocabulary_size']}")
print(f"mean weight: {best_result['weight_mean']:.6e}")
print(f"max weight: {best_result['weight_max']:.6e}")
print(f"negative entries before cleanup: {best_result['negative_entries_before_cleanup']}")
print(f"overall silhouette score: {best_result['overall_silhouette_score']:.4f}")


# =========================================================
# BAR CHART: BEST SILHOUETTE PER k
# =========================================================

best_per_k = []

for k in k_values:
    rows_k = [r for r in overall_results if r["k"] == k]
    best_k_row = max(rows_k, key=lambda r: r["overall_silhouette_score"])
    best_per_k.append(best_k_row)

plt.figure(figsize=(8, 5))

x = np.arange(len(best_per_k))
scores = [r["overall_silhouette_score"] for r in best_per_k]
labels_x = [
    f"k={r['k']}\nα={r['alpha']}\nβ={r['beta']}"
    for r in best_per_k
]

plt.bar(x, scores)

plt.xticks(x, labels_x)
plt.ylabel("Best overall Silhouette Score")
plt.title("Best SIWKL inverse-G beta=True uncapped result per k")

for i, score in enumerate(scores):
    plt.text(
        i,
        score,
        f"{score:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

bar_file = "SIWKL_inverse_G_global_beta_true_uncapped_best_per_k_bar_chart.png"
plt.savefig(bar_file, dpi=300)
plt.show()

print(f"Saved bar chart: {bar_file}")