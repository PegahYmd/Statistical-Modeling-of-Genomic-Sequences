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

beta_values = [0.001, 0.01, 0.1, 1.0]

fasta_pattern = "../1-samples/*.fa"

method_name = "SIWKL_inverse_sqrt_G_beta_true_uncapped"


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
# K-MER COUNTER
# =========================================================

def count_kmers(sequence, k):
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
    w(x) = 1 / sqrt(G(x))

    This is a KL-based dissimilarity score, not a formal metric.
    """
    return np.sum(w * P * np.log(P / Q))


# =========================================================
# SAMPLE STATUS
# =========================================================

def classify_sample_status(score):
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
# READ SEQUENCES
# =========================================================

sequences = [read_fasta(f) for f in files]

print("\nFinished reading FASTA files.")


# =========================================================
# CREATE LABELS FROM FILE NAMES
# =========================================================

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


# =========================================================
# RESULTS STORAGE
# =========================================================

overall_results = []
all_per_sample_results = []


# =========================================================
# MAIN LOOP
# =========================================================

for k in k_values:

    print("\n" + "=" * 100)
    print(f"Preparing k-mer matrix for k = {k}")
    print("=" * 100)

    # =====================================================
    # COUNT K-MERS
    # =====================================================

    all_counts = []

    for sequence in sequences:
        counts = count_kmers(sequence, k)
        all_counts.append(counts)

    # =====================================================
    # BUILD GLOBAL VOCABULARY
    # =====================================================

    vocab = sorted(set().union(*[set(c.keys()) for c in all_counts]))
    vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}

    V = len(vocab)

    print(f"Vocabulary size for k={k}: {V}")

    # =====================================================
    # BUILD COUNT MATRIX
    # =====================================================

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

    print(f"Zero percentage before smoothing: {zero_percentage:.4f}%")

    # =====================================================
    # GLOBAL COUNTS
    # =====================================================

    global_counts = count_matrix.sum(axis=0)
    global_total = global_counts.sum()

    if np.any(global_counts == 0):
        raise ValueError("Unexpected zero global count found in vocabulary.")

    # =====================================================
    # LOOP OVER ALPHA AND BETA
    # =====================================================

    for alpha in alpha_values:

        # -------------------------------------------------
        # Sample-level smoothing
        # P_i(x) = (c_i(x) + alpha) /
        #          (sum_y c_i(y) + alpha V)
        # -------------------------------------------------

        P = (count_matrix + alpha) / (row_sums + alpha * V)

        zero_probs_after_alpha = np.sum(P == 0)

        if zero_probs_after_alpha != 0:
            raise ValueError("Zero probabilities remain after alpha smoothing.")

        for beta in beta_values:

            print("\n" + "-" * 90)
            print(f"Running k={k}, alpha={alpha}, beta={beta}, uncapped=True")
            print("-" * 90)

            # =================================================
            # GLOBAL DISTRIBUTION WITH BETA SMOOTHING
            # =================================================
            #
            # G(x) = (C(x) + beta) /
            #        (sum_y C(y) + beta V)

            G = (global_counts + beta) / (global_total + beta * V)

            if np.any(G <= 0):
                raise ValueError("G(x) contains zero or negative values.")

            # =================================================
            # UNCAPPED WEIGHT
            # =================================================
            #
            # w(x) = 1 / sqrt(G(x))
            #
            # No cap is applied.

            w = 1.0 / np.sqrt(G)

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
            # SYMMETRIC MATRIX
            # =================================================

            D_symmetric = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                for j in range(N):
                    D_symmetric[i, j] = 0.5 * (
                        D_directed[i, j] + D_directed[j, i]
                    )

            np.fill_diagonal(D_symmetric, 0)

            # =================================================
            # CHECK NEGATIVE VALUES
            # =================================================

            negative_entries = np.sum(D_symmetric < 0)
            min_distance = np.min(D_symmetric)
            max_distance = np.max(D_symmetric)

            print("Distance statistics:")
            print(f"  min symmetric score: {min_distance:.6e}")
            print(f"  max symmetric score: {max_distance:.6e}")
            print(f"  negative entries:    {negative_entries}")

            # For silhouette_score, distances must be non-negative.
            D_for_silhouette = D_symmetric.copy()

            if negative_entries > 0:
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

            for name, class_name, sample_score in zip(
                sample_names,
                label_names,
                sample_silhouettes
            ):
                status = classify_sample_status(sample_score)

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
# SAVE OVERALL RESULTS CSV
# =========================================================

overall_file = "SIWKL_inverse_sqrt_G_beta_true_uncapped_results.csv"

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

per_sample_file = "SIWKL_inverse_sqrt_G_beta_true_uncapped_per_sample_silhouette.csv"

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
# PRINT BEST SETTING
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
# BAR CHART: BEST RESULT PER k
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
plt.ylabel("Best Overall Silhouette Score")
plt.title("SIWKL inverse sqrt-G, beta=True, uncapped")

for i, score in enumerate(scores):
    plt.text(
        i,
        score,
        f"{score:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

bar_file = "SIWKL_inverse_sqrt_G_beta_true_uncapped_best_per_k_bar_chart.png"
plt.savefig(bar_file, dpi=300)
plt.show()

print(f"Saved bar chart: {bar_file}")