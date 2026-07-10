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

fasta_pattern = "../1-samples/*.fa"

method_name = "baseline_KL_w1"

# If True, it creates one per-sample bar chart for every k-alpha setting.
# If False, it creates only one per-sample bar chart for the best setting.
plot_all_per_sample_bars = False


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
# KL DIVERGENCE
# =========================================================

def kl_divergence(P, Q):
    """
    KL(P || Q) = sum_x P(x) log(P(x) / Q(x))
    """
    return np.sum(P * np.log(P / Q))


# =========================================================
# SAMPLE STATUS BASED ON PER-SAMPLE SILHOUETTE
# =========================================================

def classify_sample_status(score):
    """
    Interprets each per-sample silhouette score.
    """
    if score > 0.10:
        return "clear"
    elif score >= 0:
        return "weak/ambiguous"
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
# MAIN LOOP OVER k AND alpha
# =========================================================

for k in k_values:

    print("\n" + "=" * 90)
    print(f"Preparing k-mer count matrix for k = {k}")
    print("=" * 90)

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
    # LOOP OVER alpha VALUES
    # =====================================================

    for alpha in alpha_values:

        print("\n" + "-" * 70)
        print(f"Running baseline KL with w(x)=1, k={k}, alpha={alpha}")
        print("-" * 70)

        # =================================================
        # SMOOTHED PROBABILITY DISTRIBUTIONS
        # =================================================
        # Baseline KL means w(x)=1.
        # There is no beta here.
        #
        # P_i(x) = (c_i(x) + alpha) / (sum_y c_i(y) + alpha V)

        P = (count_matrix + alpha) / (row_sums + alpha * V)

        row_sum_check = P.sum(axis=1)
        zero_probs_after = np.sum(P == 0)

        print("Probability row sums:")
        print(np.round(row_sum_check, 6))
        print("Zero probabilities after smoothing:", zero_probs_after)

        # =================================================
        # DIRECTED KL MATRIX
        # =================================================

        D_kl_directed = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i != j:
                    D_kl_directed[i, j] = kl_divergence(P[i], P[j])

        # =================================================
        # SYMMETRIC KL MATRIX
        # =================================================
        # Used for Silhouette Score.

        D_kl_symmetric = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                D_kl_symmetric[i, j] = 0.5 * (
                    D_kl_directed[i, j] + D_kl_directed[j, i]
                )

        np.fill_diagonal(D_kl_symmetric, 0)

        # Numerical cleanup
        D_kl_symmetric[D_kl_symmetric < 0] = 0

        # =================================================
        # OVERALL SILHOUETTE SCORE
        # =================================================

        overall_sil = silhouette_score(
            D_kl_symmetric,
            labels,
            metric="precomputed"
        )

        print(f"Overall Silhouette Score: {overall_sil:.4f}")

        # =================================================
        # PER-SAMPLE SILHOUETTE SCORES
        # =================================================

        sample_silhouettes = silhouette_samples(
            D_kl_symmetric,
            labels,
            metric="precomputed"
        )

        print("\nPer-sample Silhouette Scores:")
        print("-" * 100)
        print(f"{'Sample':35s} {'Class':15s} {'Score':>12s} {'Status':>22s}")
        print("-" * 100)

        for name, class_name, sample_score in zip(sample_names, label_names, sample_silhouettes):
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
                "beta": "NA",
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
            "beta": "NA",
            "vocabulary_size": V,
            "zero_percentage_before_smoothing": zero_percentage,
            "zero_probs_after_smoothing": zero_probs_after,
            "overall_silhouette_score": overall_sil
        })

        # =================================================
        # OPTIONAL PER-SAMPLE BAR CHART FOR EVERY SETTING
        # =================================================

        if plot_all_per_sample_bars:
            x = np.arange(N)

            plt.figure(figsize=(10, 5))
            plt.bar(x, sample_silhouettes)
            plt.axhline(0, linestyle="--", linewidth=1)

            plt.xticks(x, sample_names, rotation=90)
            plt.ylabel("Per-sample Silhouette Score")
            plt.title(f"Per-sample Silhouette: {method_name}, k={k}, alpha={alpha}")

            for idx, score in enumerate(sample_silhouettes):
                plt.text(
                    idx,
                    score,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom" if score >= 0 else "top",
                    fontsize=8,
                    rotation=90
                )

            plt.tight_layout()

            fig_name = f"per_sample_silhouette_{method_name}_k{k}_alpha{alpha}.png"
            plt.savefig(fig_name, dpi=300)
            plt.show()

            print(f"Saved per-sample bar chart: {fig_name}")


# =========================================================
# PRINT OVERALL RESULTS TABLE
# =========================================================

print("\n" + "=" * 110)
print("FULL RESULTS TABLE: BASELINE KL WITH w(x)=1")
print("=" * 110)

print(
    f"{'Method':<18} "
    f"{'k':>5} "
    f"{'alpha':>10} "
    f"{'beta':>8} "
    f"{'Vocab size':>15} "
    f"{'Zeros before (%)':>18} "
    f"{'Overall Silhouette':>20}"
)

print("-" * 110)

for r in overall_results:
    print(
        f"{r['method']:<18} "
        f"{r['k']:>5} "
        f"{r['alpha']:>10.3g} "
        f"{r['beta']:>8} "
        f"{r['vocabulary_size']:>15} "
        f"{r['zero_percentage_before_smoothing']:>18.4f} "
        f"{r['overall_silhouette_score']:>20.4f}"
    )


# =========================================================
# SAVE OVERALL RESULTS CSV
# =========================================================

overall_file = "baseline_KL_w1_overall_silhouette_summary.csv"

with open(overall_file, "w") as f:
    f.write(
        "method,k,alpha,beta,vocabulary_size,"
        "zero_percentage_before_smoothing,"
        "zero_probs_after_smoothing,"
        "overall_silhouette_score\n"
    )

    for r in overall_results:
        f.write(
            f"{r['method']},"
            f"{r['k']},"
            f"{r['alpha']},"
            f"{r['beta']},"
            f"{r['vocabulary_size']},"
            f"{r['zero_percentage_before_smoothing']:.8f},"
            f"{r['zero_probs_after_smoothing']},"
            f"{r['overall_silhouette_score']:.8f}\n"
        )

print(f"\nSaved overall results CSV: {overall_file}")


# =========================================================
# SAVE PER-SAMPLE RESULTS CSV
# =========================================================

per_sample_file = "baseline_KL_w1_per_sample_silhouette_scores.csv"

with open(per_sample_file, "w") as f:
    f.write(
        "method,k,alpha,beta,sample,class,"
        "sample_silhouette_score,status\n"
    )

    for r in all_per_sample_results:
        f.write(
            f"{r['method']},"
            f"{r['k']},"
            f"{r['alpha']},"
            f"{r['beta']},"
            f"{r['sample']},"
            f"{r['class']},"
            f"{r['sample_silhouette_score']:.8f},"
            f"{r['status']}\n"
        )

print(f"Saved per-sample results CSV: {per_sample_file}")


# =========================================================
# GROUPED BAR CHART: OVERALL SILHOUETTE SCORES
# =========================================================

score_matrix = np.zeros((len(alpha_values), len(k_values)))

for a_idx, alpha in enumerate(alpha_values):
    for k_idx, k in enumerate(k_values):
        for r in overall_results:
            if r["k"] == k and r["alpha"] == alpha:
                score_matrix[a_idx, k_idx] = r["overall_silhouette_score"]

x = np.arange(len(k_values))
bar_width = 0.18

plt.figure(figsize=(9, 5))

for a_idx, alpha in enumerate(alpha_values):
    offset = (a_idx - (len(alpha_values) - 1) / 2) * bar_width

    bars = plt.bar(
        x + offset,
        score_matrix[a_idx],
        width=bar_width,
        label=f"alpha={alpha}"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

plt.xlabel("k-mer size")
plt.ylabel("Overall Silhouette Score")
plt.title("Baseline KL with w(x)=1: sensitivity to k and alpha")
plt.xticks(x, [str(k) for k in k_values])
plt.legend()
plt.tight_layout()

overall_bar_file = "baseline_KL_w1_overall_silhouette_bar_chart.png"
plt.savefig(overall_bar_file, dpi=300)
plt.show()

print(f"Saved overall bar chart: {overall_bar_file}")


# =========================================================
# FIND BEST SETTING
# =========================================================

best_result = max(
    overall_results,
    key=lambda r: r["overall_silhouette_score"]
)

best_k = best_result["k"]
best_alpha = best_result["alpha"]
best_score = best_result["overall_silhouette_score"]

print("\n" + "=" * 80)
print("BEST SETTING")
print("=" * 80)
print(f"Best k: {best_k}")
print(f"Best alpha: {best_alpha}")
print(f"Best overall Silhouette Score: {best_score:.4f}")


# =========================================================
# PER-SAMPLE BAR CHART FOR BEST SETTING
# =========================================================

best_sample_rows = [
    r for r in all_per_sample_results
    if r["k"] == best_k and r["alpha"] == best_alpha
]

best_sample_names = [r["sample"] for r in best_sample_rows]
best_sample_scores = np.array([r["sample_silhouette_score"] for r in best_sample_rows])
best_sample_status = [r["status"] for r in best_sample_rows]

x = np.arange(len(best_sample_names))

plt.figure(figsize=(10, 5))
plt.bar(x, best_sample_scores)
plt.axhline(0, linestyle="--", linewidth=1)

plt.xticks(x, best_sample_names, rotation=90)
plt.ylabel("Per-sample Silhouette Score")
plt.title(
    f"Per-sample Silhouette Scores for Best Setting\n"
    f"{method_name}, k={best_k}, alpha={best_alpha}, overall={best_score:.4f}"
)

for idx, score in enumerate(best_sample_scores):
    plt.text(
        idx,
        score,
        f"{score:.3f}",
        ha="center",
        va="bottom" if score >= 0 else "top",
        fontsize=8,
        rotation=90
    )

plt.tight_layout()

best_bar_file = f"best_setting_per_sample_silhouette_{method_name}_k{best_k}_alpha{best_alpha}.png"
plt.savefig(best_bar_file, dpi=300)
plt.show()

print(f"Saved best-setting per-sample bar chart: {best_bar_file}")


# =========================================================
# PRINT BEST SETTING PER-SAMPLE TABLE
# =========================================================

print("\n" + "=" * 100)
print("PER-SAMPLE SCORES FOR BEST SETTING")
print("=" * 100)
print(f"Best setting: k={best_k}, alpha={best_alpha}, overall silhouette={best_score:.4f}")
print("-" * 100)
print(f"{'Sample':35s} {'Class':15s} {'Score':>12s} {'Status':>22s}")
print("-" * 100)

for r in best_sample_rows:
    print(
        f"{r['sample']:35s} "
        f"{r['class']:15s} "
        f"{r['sample_silhouette_score']:12.4f} "
        f"{r['status']:22s}"
    )