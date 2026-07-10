import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
import time
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples


# =========================================================
# PARAMETERS
# =========================================================

k_values      = [15, 21, 31]
alpha_values  = [0.001, 0.01, 0.1, 1.0]
fasta_pattern = "../1-samples/*.fa"
method_name   = "baseline_KL_w1"

# Top-N most frequent k-mers kept per genome
# Keeps memory flat regardless of k or number of genomes
TOP_K_PER_GENOME = 100_000


# =========================================================
# HELPERS
# =========================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_fasta(path):
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line.upper())
    return "".join(seq)


def count_kmers_fast(sequence, k):
    """
    Fast k-mer counting using numpy cumsum trick.
    Only valid ACGT k-mers are counted.
    """
    n = len(sequence)
    if n < k:
        return Counter()
    seq_bytes  = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    valid      = np.isin(seq_bytes, [65, 67, 71, 84])   # A=65 C=67 G=71 T=84
    cum        = np.concatenate(([0], np.cumsum(valid)))
    win_sums   = cum[k:] - cum[:n - k + 1]
    valid_pos  = np.where(win_sums == k)[0]
    return Counter(sequence[i: i + k] for i in valid_pos)


def kl_matrix_vectorized(P):
    """
    Symmetric KL divergence matrix via single matrix multiply.
    KL(i||j) = sum_x P_i(x) * (log P_i(x) - log P_j(x))
             = H_self[i] - (P @ logP.T)[i,j]
    """
    log_P  = np.log(P)
    H_self = np.sum(P * log_P, axis=1)
    cross  = P @ log_P.T
    D      = H_self[:, np.newaxis] - cross
    np.fill_diagonal(D, 0.0)
    D      = np.maximum(D, 0.0)
    D_sym  = 0.5 * (D + D.T)
    np.fill_diagonal(D_sym, 0.0)
    return D_sym


def classify_sample_status(score):
    if score > 0.10: return "clear"
    elif score >= 0: return "weak/ambiguous"
    else:            return "possibly_misplaced"


def silhouette_named_only(D_sym, labels):
    """
    Computes silhouette score restricted to named species
    (excludes Brucella_sp catch-all group = label 7)
    and only for classes with >= 2 members.
    Returns score or None if not computable.
    """
    named_mask = labels != 7
    if named_mask.sum() < 2:
        return None
    named_labels = labels[named_mask]
    named_D      = D_sym[np.ix_(named_mask, named_mask)]

    from collections import Counter as C
    valid_classes = {cls for cls, cnt in C(named_labels).items() if cnt >= 2}
    valid_mask    = np.array([lab in valid_classes for lab in named_labels])

    if valid_mask.sum() < 2 or len(set(named_labels[valid_mask])) < 2:
        return None

    return silhouette_score(
        named_D[np.ix_(valid_mask, valid_mask)],
        named_labels[valid_mask],
        metric="precomputed"
    )


# =========================================================
# LOAD FILES
# =========================================================

files = sorted(glob.glob(fasta_pattern))
if not files:
    raise FileNotFoundError(f"No FASTA files found: {fasta_pattern}")

sample_names = [os.path.basename(f) for f in files]
N            = len(files)
log(f"Found {N} FASTA files.")


# =========================================================
# READ SEQUENCES ONCE (reused across all k and alpha)
# =========================================================

log("Reading sequences...")
sequences = []
for i, f in enumerate(files):
    seq = read_fasta(f)
    sequences.append(seq)
    log(f"  [{i+1}/{N}] {os.path.basename(f)}  len={len(seq):,}")
log("Done reading.")


# =========================================================
# LABELS
# =========================================================

species_map = {
    "GCA_000740135.1": 0, "GCA_000740155.1": 0,   # abortus
    "GCA_001307475.2": 1,                           # melitensis
    "GCA_000740235.1": 2,                           # suis
    "GCA_000691585.1": 3,                           # canis
    "GCA_000740275.1": 4,                           # pinnipedialis
    "GCA_900000005.1": 5,                           # vulpis
    "GCA_900095155.1": 6,                           # inopinata
    "GCA_000371045.1": 7, "GCA_000370965.1": 7,    # Brucella sp.
    "GCA_000371025.1": 7, "GCA_000371065.1": 7,
    "GCA_000371005.1": 7, "GCA_000158995.1": 7,
    "GCA_000163135.1": 7, "GCA_000157875.1": 7,
    "GCA_015832115.1": 7, "GCA_001971625.1": 7,
    "GCA_900092405.1": 7, "GCA_000370945.1": 7,
    "GCA_000370985.1": 7, "GCA_000370925.1": 7,
    "GCA_000480195.1": 7, "GCA_001971805.1": 7,
    "GCA_001742815.1": 7, "GCA_001715385.1": 7,
    "GCA_009601725.1": 7, "GCA_014084005.1": 7,
    "GCA_014495905.1": 7,
}

label_name_map = {
    0: "abortus",      1: "melitensis",    2: "suis",
    3: "canis",        4: "pinnipedialis", 5: "vulpis",
    6: "inopinata",    7: "Brucella_sp"
}

labels = []
for name in sample_names:
    acc = "_".join(name.split("_")[:2])
    if acc not in species_map:
        raise ValueError(f"Unknown accession '{acc}' in '{name}'")
    labels.append(species_map[acc])

labels      = np.array(labels)
label_names = [label_name_map[lab] for lab in labels]

log("Labels:")
for name, lname in zip(sample_names, label_names):
    log(f"  {name:50s}  {lname}")


# =========================================================
# RESULTS STORAGE
# =========================================================

overall_results        = []
all_per_sample_results = []


# =========================================================
# OUTER LOOP: k values
# =========================================================

for k in k_values:

    log("=" * 70)
    log(f"K = {k}")
    log("=" * 70)

    # ---------------------------------------------------
    # COUNT K-MERS (once per k, reused across all alpha)
    # ---------------------------------------------------
    log(f"Counting k-mers (top {TOP_K_PER_GENOME:,} per genome)...")
    corpus_vocab      = Counter()
    per_genome_counts = []

    for i, seq in enumerate(sequences):
        t0         = time.time()
        counts     = count_kmers_fast(seq, k)
        top_counts = Counter(dict(counts.most_common(TOP_K_PER_GENOME)))
        corpus_vocab.update(top_counts)
        per_genome_counts.append(top_counts)
        log(f"  [{i+1}/{N}] {sample_names[i]:50s}  "
            f"unique={len(counts):>8,}  kept={len(top_counts):,}  "
            f"({time.time()-t0:.1f}s)")
        del counts

    # Shared vocabulary: top-N across corpus
    vocab       = [kmer for kmer, _ in
                   corpus_vocab.most_common(TOP_K_PER_GENOME)]
    vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}
    V           = len(vocab)
    log(f"Shared vocabulary size: {V:,}")
    del corpus_vocab

    # ---------------------------------------------------
    # BUILD COUNT MATRIX (once per k)
    # ---------------------------------------------------
    log("Building count matrix...")
    count_matrix = np.zeros((N, V), dtype=np.float32)
    for i, top_counts in enumerate(per_genome_counts):
        for kmer, cnt in top_counts.items():
            j = vocab_index.get(kmer)
            if j is not None:
                count_matrix[i, j] = cnt
    del per_genome_counts, vocab_index

    row_sums = count_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError(f"Zero k-mers in at least one sample at k={k}.")

    zero_pct = 100.0 * np.sum(count_matrix == 0) / count_matrix.size
    log(f"Zero entries before smoothing: {zero_pct:.2f}%")

    # ---------------------------------------------------
    # INNER LOOP: alpha values
    # (count matrix is reused — only P changes)
    # ---------------------------------------------------

    for alpha in alpha_values:

        log("-" * 60)
        log(f"  alpha = {alpha}  (k={k})")
        log("-" * 60)

        # Smoothed probability matrix
        P = (count_matrix.astype(np.float64) + alpha) / \
            (row_sums.astype(np.float64) + alpha * V)
        log(f"  Row sum range: "
            f"{P.sum(axis=1).min():.6f} – {P.sum(axis=1).max():.6f}")

        # Vectorized KL matrix
        log("  Computing KL distance matrix...")
        t0    = time.time()
        D_sym = kl_matrix_vectorized(P)
        del P
        log(f"  KL matrix done in {time.time()-t0:.2f}s")

        # Full silhouette (all 29 samples, all 8 label groups)
        overall_sil = silhouette_score(
            D_sym, labels, metric="precomputed"
        )

        # Named-species-only silhouette
        named_sil = silhouette_named_only(D_sym, labels)

        log(f"  Overall Silhouette (all):          {overall_sil:.4f}")
        log(f"  Silhouette (named species only):   "
            f"{named_sil:.4f}" if named_sil is not None else "  N/A")

        # Per-sample silhouette
        sample_sils = silhouette_samples(
            D_sym, labels, metric="precomputed"
        )

        log(f"\n  {'Sample':50s} {'Class':15s} {'Score':>10s}  Status")
        log("  " + "-" * 90)
        for name, lname, sc in zip(sample_names, label_names, sample_sils):
            status = classify_sample_status(sc)
            log(f"  {name:50s} {lname:15s} {sc:10.4f}  {status}")
            all_per_sample_results.append({
                "method": method_name,
                "k": k,
                "alpha": alpha,
                "beta": "NA",
                "sample": name,
                "class": lname,
                "sample_silhouette_score": sc,
                "status": status
            })

        overall_results.append({
            "method": method_name,
            "k": k,
            "alpha": alpha,
            "beta": "NA",
            "vocabulary_size": V,
            "zero_pct": zero_pct,
            "overall_silhouette_score": overall_sil,
            "named_silhouette_score": named_sil if named_sil is not None else float("nan")
        })

        del D_sym

    # Free count matrix before next k
    del count_matrix, row_sums


# =========================================================
# SUMMARY TABLE
# =========================================================

log("\n" + "=" * 90)
log("FULL RESULTS TABLE")
log("=" * 90)
log(f"{'k':>5}  {'alpha':>8}  {'Vocab':>10}  "
    f"{'Zeros%':>8}  {'Sil (all)':>12}  {'Sil (named)':>13}")
log("-" * 65)
for r in overall_results:
    named_str = f"{r['named_silhouette_score']:13.4f}" \
        if not np.isnan(r['named_silhouette_score']) else "          N/A"
    log(f"{r['k']:>5}  {r['alpha']:>8.4g}  {r['vocabulary_size']:>10,}  "
        f"{r['zero_pct']:>8.2f}  {r['overall_silhouette_score']:>12.4f}  "
        f"{named_str}")


# =========================================================
# SAVE CSVs
# =========================================================

overall_file = f"{method_name}_overall_silhouette_summary.csv"
with open(overall_file, "w") as f:
    f.write("method,k,alpha,beta,vocabulary_size,zero_pct,"
            "overall_silhouette_score,named_silhouette_score\n")
    for r in overall_results:
        f.write(f"{r['method']},{r['k']},{r['alpha']},{r['beta']},"
                f"{r['vocabulary_size']},{r['zero_pct']:.4f},"
                f"{r['overall_silhouette_score']:.8f},"
                f"{r['named_silhouette_score']:.8f}\n")
log(f"Saved: {overall_file}")

per_sample_file = f"{method_name}_per_sample_silhouette_scores.csv"
with open(per_sample_file, "w") as f:
    f.write("method,k,alpha,beta,sample,class,"
            "sample_silhouette_score,status\n")
    for r in all_per_sample_results:
        f.write(f"{r['method']},{r['k']},{r['alpha']},{r['beta']},"
                f"{r['sample']},{r['class']},"
                f"{r['sample_silhouette_score']:.8f},{r['status']}\n")
log(f"Saved: {per_sample_file}")


# =========================================================
# GROUPED BAR CHART: overall silhouette by k and alpha
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (score_key, title_suffix) in enumerate([
    ("overall_silhouette_score", "All 29 samples"),
    ("named_silhouette_score",   "Named species only")
]):
    ax = axes[ax_idx]
    score_matrix = np.zeros((len(alpha_values), len(k_values)))
    for a_idx, alpha in enumerate(alpha_values):
        for k_idx, k in enumerate(k_values):
            for r in overall_results:
                if r["k"] == k and r["alpha"] == alpha:
                    val = r[score_key]
                    score_matrix[a_idx, k_idx] = 0.0 if np.isnan(val) else val

    x         = np.arange(len(k_values))
    bar_width = 0.18
    for a_idx, alpha in enumerate(alpha_values):
        offset = (a_idx - (len(alpha_values) - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, score_matrix[a_idx],
                        width=bar_width, label=f"alpha={alpha}")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    ax.set_xlabel("k-mer size")
    ax.set_ylabel("Overall Silhouette Score")
    ax.set_title(f"{method_name}\nSensitivity to k and alpha — {title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.tight_layout()
bar_chart_file = f"{method_name}_overall_silhouette_bar_chart.png"
plt.savefig(bar_chart_file, dpi=200)
plt.close()
log(f"Saved: {bar_chart_file}")


# =========================================================
# PER-SAMPLE BAR CHARTS FOR BEST SETTING
# =========================================================

# Best by overall score
best_overall = max(overall_results,
                   key=lambda r: r["overall_silhouette_score"])
# Best by named score
valid_named  = [r for r in overall_results
                if not np.isnan(r["named_silhouette_score"])]
best_named   = max(valid_named,
                   key=lambda r: r["named_silhouette_score"]) \
               if valid_named else best_overall

for best_r, tag in [(best_overall, "best_overall"),
                    (best_named,   "best_named")]:

    best_k     = best_r["k"]
    best_alpha = best_r["alpha"]
    best_score = best_r["overall_silhouette_score"]

    rows = [r for r in all_per_sample_results
            if r["k"] == best_k and r["alpha"] == best_alpha]
    snames  = [r["sample"] for r in rows]
    scores  = np.array([r["sample_silhouette_score"] for r in rows])
    classes = [r["class"] for r in rows]

    x = np.arange(len(snames))
    plt.figure(figsize=(16, 5))
    plt.bar(x, scores)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, snames, rotation=90, fontsize=6)
    plt.ylabel("Per-sample Silhouette Score")
    plt.title(
        f"Per-sample Silhouette — {tag}\n"
        f"{method_name}  k={best_k}  alpha={best_alpha}  "
        f"overall={best_score:.4f}"
    )
    for idx, sc in enumerate(scores):
        plt.text(idx, sc, f"{sc:.3f}", ha="center",
                 va="bottom" if sc >= 0 else "top",
                 fontsize=5, rotation=90)
    plt.tight_layout()
    fig_name = f"per_sample_silhouette_{method_name}_{tag}_k{best_k}_alpha{best_alpha}.png"
    plt.savefig(fig_name, dpi=150)
    plt.close()
    log(f"Saved: {fig_name}")

log("\nALL DONE.")