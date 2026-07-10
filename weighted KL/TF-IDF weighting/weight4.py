import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import Counter

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# =========================================================
# PARAMETERS
# =========================================================

k_values = [15, 21, 31]

fasta_pattern = "../1-samples/*.fa"

epsilon = 1e-12

use_smooth_idf = True


# =========================================================
# FASTA READER
# =========================================================

def read_fasta(path):
    """
    Reads a FASTA file and returns the sequence as one uppercase string.
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
    Counts valid DNA k-mers in one sequence.
    Only k-mers containing A, C, G, T are used.
    """
    counts = Counter()
    valid_chars = set("ACGT")

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]

        if set(kmer).issubset(valid_chars):
            counts[kmer] += 1

    return counts


# =========================================================
# KL DIVERGENCE FUNCTION
# =========================================================

def kl_divergence(P, Q):
    """
    KL(P || Q) = sum_x P(x) log(P(x) / Q(x))
    """
    return np.sum(P * np.log(P / Q))


# =========================================================
# HEATMAP FUNCTION
# =========================================================

def plot_heatmap(matrix, sample_names, title, output_name):
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrix,
        xticklabels=sample_names,
        yticklabels=sample_names,
        cmap="viridis",
        annot=True,
        fmt=".3f"
    )

    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()


# =========================================================
# CLUSTERMAP FUNCTION
# =========================================================

def plot_clustermap_from_distance(matrix, sample_names, title, output_name):
    """
    Creates a clustermap using a precomputed distance matrix.
    """
    condensed = squareform(matrix, checks=False)
    Z = linkage(condensed, method="average")

    sns.clustermap(
        matrix,
        row_linkage=Z,
        col_linkage=Z,
        xticklabels=sample_names,
        yticklabels=sample_names,
        cmap="viridis",
        figsize=(10, 10)
    )

    plt.suptitle(title, y=1.02)
    plt.savefig(output_name, dpi=300)
    plt.show()


# =========================================================
# LOAD FASTA FILES ONCE
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

sequences = []

for f in files:
    sequences.append(read_fasta(f))

print("\nFinished reading FASTA files.")


# =========================================================
# CREATE LABELS FOR SILHOUETTE SCORE
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


# =========================================================
# STORE SILHOUETTE RESULTS
# =========================================================

silhouette_results = []


# =========================================================
# MAIN LOOP OVER k VALUES
# =========================================================

for k in k_values:

    print("\n" + "=" * 70)
    print(f"Running TF-IDF normalized KL for k = {k}")
    print("=" * 70)

    # =====================================================
    # COUNT K-MERS FOR EACH SAMPLE
    # =====================================================

    all_counts = []

    for sequence in sequences:
        counts = count_kmers(sequence, k)
        all_counts.append(counts)

    print("Finished counting k-mers.")

    # =====================================================
    # BUILD GLOBAL K-MER VOCABULARY
    # =====================================================
    # The vocabulary contains every unique k-mer appearing
    # in the whole FASTA dataset for this k.

    vocab = sorted(set().union(*[set(c.keys()) for c in all_counts]))
    vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}

    V = len(vocab)

    print(f"Global vocabulary size for k={k}: {V}")

    # =====================================================
    # BUILD COUNT MATRIX
    # =====================================================
    # count_matrix[i, j] = count of k-mer j in sample i

    count_matrix = np.zeros((N, V), dtype=np.float64)

    for i, counts in enumerate(all_counts):
        for kmer, count in counts.items():
            j = vocab_index[kmer]
            count_matrix[i, j] = count

    # =====================================================
    # COMPUTE TERM FREQUENCY: TF
    # =====================================================
    # TF(x, S_i) = count_i(x) / total number of k-mers in S_i

    row_sums = count_matrix.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0):
        raise ValueError(f"At least one FASTA file produced zero valid k-mers for k={k}.")

    tf_matrix = count_matrix / row_sums

    # =====================================================
    # COMPUTE DOCUMENT FREQUENCY: DF
    # =====================================================
    # df(x) = number of FASTA samples where k-mer x appears

    df = np.sum(count_matrix > 0, axis=0)

    print("DF statistics:")
    print("  min df:", np.min(df))
    print("  max df:", np.max(df))
    print("  mean df:", np.mean(df))

    # =====================================================
    # COMPUTE INVERSE DOCUMENT FREQUENCY: IDF
    # =====================================================
    # Smoothed IDF:
    # IDF(x) = log((N + 1) / (df(x) + 1)) + 1

    if use_smooth_idf:
        idf = np.log((N + 1) / (df + 1)) + 1
        idf_name = "smooth"
    else:
        idf = np.log(N / df)
        idf_name = "strict"

    print("IDF statistics:")
    print("  min idf:", np.min(idf))
    print("  max idf:", np.max(idf))
    print("  mean idf:", np.mean(idf))

    # =====================================================
    # COMPUTE TF-IDF MATRIX
    # =====================================================
    # TFIDF(x, S_i) = TF(x, S_i) * IDF(x)

    tfidf_matrix = tf_matrix * idf

    # =====================================================
    # NORMALIZE TF-IDF VECTORS
    # =====================================================
    # KL requires probability distributions.
    # Therefore, each TF-IDF vector is normalized to sum to 1:
    #
    # P_hat_i(x) = TFIDF(x, S_i) / sum_y TFIDF(y, S_i)

    tfidf_sums = tfidf_matrix.sum(axis=1, keepdims=True)

    if np.any(tfidf_sums == 0):
        raise ValueError(f"At least one TF-IDF vector has zero sum for k={k}.")

    P_tfidf = tfidf_matrix / tfidf_sums

    # Add small smoothing to avoid log(0)
    P_tfidf = P_tfidf + epsilon
    P_tfidf = P_tfidf / P_tfidf.sum(axis=1, keepdims=True)

    print("Check probability row sums:")
    print(P_tfidf.sum(axis=1))

    # =====================================================
    # COMPUTE KL MATRICES
    # =====================================================

    D_kl_directed = np.zeros((N, N), dtype=np.float64)
    D_kl_symmetric = np.zeros((N, N), dtype=np.float64)

    # Directed KL:
    # D_kl_directed[i, j] = KL(P_i || P_j)
    for i in range(N):
        for j in range(N):
            if i != j:
                D_kl_directed[i, j] = kl_divergence(P_tfidf[i], P_tfidf[j])

    # Symmetric KL:
    # D_sym(P_i, P_j) = 1/2 [KL(P_i || P_j) + KL(P_j || P_i)]
    for i in range(N):
        for j in range(N):
            D_kl_symmetric[i, j] = 0.5 * (
                D_kl_directed[i, j] + D_kl_directed[j, i]
            )

    np.fill_diagonal(D_kl_directed, 0)
    np.fill_diagonal(D_kl_symmetric, 0)

    # =====================================================
    # SAVE MATRICES
    # =====================================================

    directed_csv = f"tfidf_normalized_KL_directed_{idf_name}_idf_k{k}.csv"
    symmetric_csv = f"tfidf_normalized_KL_symmetric_{idf_name}_idf_k{k}.csv"

    np.savetxt(
        directed_csv,
        D_kl_directed,
        delimiter=",",
        fmt="%.8f"
    )

    np.savetxt(
        symmetric_csv,
        D_kl_symmetric,
        delimiter=",",
        fmt="%.8f"
    )

    print("Saved matrices:")
    print(" ", directed_csv)
    print(" ", symmetric_csv)

    # =====================================================
    # PLOT HEATMAPS
    # =====================================================

    plot_heatmap(
        D_kl_directed,
        sample_names,
        f"Directed KL on normalized TF-IDF k-mer distributions, k={k}",
        f"tfidf_normalized_KL_directed_{idf_name}_idf_heatmap_k{k}.png"
    )

    plot_heatmap(
        D_kl_symmetric,
        sample_names,
        f"Symmetric KL on normalized TF-IDF k-mer distributions, k={k}",
        f"tfidf_normalized_KL_symmetric_{idf_name}_idf_heatmap_k{k}.png"
    )

    # =====================================================
    # CLUSTERMAP USING SYMMETRIC KL
    # =====================================================
    # For clustering, use symmetric KL, not directed KL.

    plot_clustermap_from_distance(
        D_kl_symmetric,
        sample_names,
        f"Clustering: Symmetric KL on normalized TF-IDF, k={k}",
        f"tfidf_normalized_KL_symmetric_{idf_name}_idf_clustermap_k{k}.png"
    )

    # =====================================================
    # SILHOUETTE SCORE
    # =====================================================

    if np.all(labels != -1) and len(set(labels)) > 1:

        sil_kl = silhouette_score(
            D_kl_symmetric,
            labels,
            metric="precomputed"
        )

        print(f"Silhouette score for k={k}: {sil_kl:.4f}")

        silhouette_results.append({
            "k": k,
            "silhouette_score": sil_kl,
            "vocabulary_size": V,
            "idf_type": idf_name
        })

    else:
        print("Silhouette score was not computed.")
        print("Reason: labels were not detected from filenames.")


# =========================================================
# PRINT SUMMARY OF RESULTS
# =========================================================

print("\n" + "=" * 70)
print("SUMMARY OF SILHOUETTE SCORES")
print("=" * 70)

if len(silhouette_results) > 0:
    for result in silhouette_results:
        print(
            f"k={result['k']}, "
            f"IDF={result['idf_type']}, "
            f"vocab={result['vocabulary_size']}, "
            f"silhouette={result['silhouette_score']:.4f}"
        )
else:
    print("No silhouette scores were computed.")