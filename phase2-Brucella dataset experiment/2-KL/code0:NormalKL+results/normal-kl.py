"""
Baseline KL divergence (w(x)=1) on Brucella dataset.
Fixed alpha=1.0. Labels read from labels.csv.

Usage:
    python3 normal-kl-full.py <k>

Example:
    python3 normal-kl-full.py 15
    python3 normal-kl-full.py 21
    python3 normal-kl-full.py 31
"""

import sys
import csv
import glob
import os
import time
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples


# =========================================================
# COMMAND LINE
# =========================================================

if len(sys.argv) != 2:
    print("Usage: python3 normal-kl-full.py <k>")
    sys.exit(1)

k = int(sys.argv[1])


# =========================================================
# PARAMETERS
# =========================================================

alpha         = 1.0
fasta_pattern = "../1-samples/*.fa"
labels_csv    = "../labels.csv"
method_name   = "baseline_KL_w1"


# =========================================================
# HELPERS
# =========================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line.upper())
    return "".join(seq)


def count_kmers_fast(sequence, k):
    """
    Fast k-mer counting using numpy cumsum trick.
    Only counts k-mers composed entirely of A, C, G, T.
    """
    n = len(sequence)
    if n < k:
        return Counter()
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    valid     = np.isin(seq_bytes, [65, 67, 71, 84])  # A C G T
    cum       = np.concatenate(([0], np.cumsum(valid)))
    win_sums  = cum[k:] - cum[:n - k + 1]
    valid_pos = np.where(win_sums == k)[0]
    return Counter(sequence[i: i + k] for i in valid_pos)


def classify_sample_status(score):
    if score > 0.10: return "clear"
    elif score >= 0: return "weak/ambiguous"
    else:            return "possibly_misplaced"


# =========================================================
# LOAD LABELS
# =========================================================

log(f"Reading labels from {labels_csv}...")
label_lookup = {}
with open(labels_csv, newline="") as f:
    for row in csv.DictReader(f):
        label_lookup[row["sample"]] = (row["group"], int(row["group_id"]))
log(f"Loaded {len(label_lookup)} labels.")


# =========================================================
# LOAD FASTA FILES
# =========================================================

files = sorted(glob.glob(fasta_pattern))
if not files:
    raise FileNotFoundError(f"No FASTA files: {fasta_pattern}")

sample_names = [os.path.basename(f) for f in files]
N            = len(files)
log(f"Found {N} FASTA files.")

log("Reading sequences...")
sequences = []
for i, f in enumerate(files):
    seq = read_fasta(f)
    sequences.append(seq)
    log(f"  [{i+1}/{N}] {os.path.basename(f):50s}  len={len(seq):,}")
log("Done reading.")


# =========================================================
# ASSIGN LABELS
# =========================================================

labels      = []
label_names = []
for name in sample_names:
    if name not in label_lookup:
        raise ValueError(f"'{name}' not found in {labels_csv}")
    group_name, group_id = label_lookup[name]
    labels.append(group_id)
    label_names.append(group_name)

labels = np.array(labels)

log("Labels:")
for name, lname in zip(sample_names, label_names):
    log(f"  {name:50s}  {lname}")


# =========================================================
# COUNT K-MERS
# =========================================================

log(f"\nK = {k}")
log("Counting k-mers...")
all_counts = []
for i, seq in enumerate(sequences):
    t0     = time.time()
    counts = count_kmers_fast(seq, k)
    all_counts.append(counts)
    log(f"  [{i+1}/{N}] {sample_names[i]:50s}  "
        f"unique={len(counts):>8,}  ({time.time()-t0:.1f}s)")


# =========================================================
# BUILD VOCABULARY
# =========================================================

log("Building vocabulary...")
vocab       = sorted(set().union(*[c.keys() for c in all_counts]))
vocab_index = {kmer: idx for idx, kmer in enumerate(vocab)}
V           = len(vocab)
log(f"Vocabulary size: {V:,}")


# =========================================================
# BUILD COUNT MATRIX
# =========================================================

log("Building count matrix...")
count_matrix = np.zeros((N, V), dtype=np.float32)
for i, counts in enumerate(all_counts):
    for kmer, cnt in counts.items():
        count_matrix[i, vocab_index[kmer]] = cnt

del all_counts, vocab_index, vocab
gc.collect()

row_sums = count_matrix.sum(axis=1, keepdims=True)
if np.any(row_sums == 0):
    raise ValueError("Zero k-mers in at least one sample.")

zero_pct = 100.0 * np.sum(count_matrix == 0) / count_matrix.size
log(f"Count matrix: {count_matrix.shape}  "
    f"RAM ≈ {count_matrix.nbytes / 1e6:.0f} MB")
log(f"Zero entries before smoothing: {zero_pct:.2f}%")


# =========================================================
# SMOOTHED PROBABILITY MATRIX
# =========================================================

log("Computing probability matrix...")
P = (count_matrix.astype(np.float64) + alpha) / \
    (row_sums.astype(np.float64) + alpha * V)
del count_matrix, row_sums
gc.collect()
log(f"Row sum range: {P.sum(axis=1).min():.6f} – {P.sum(axis=1).max():.6f}")


# =========================================================
# VECTORIZED SYMMETRIC KL MATRIX
# =========================================================

log("Computing KL distance matrix (vectorized)...")
t0     = time.time()
log_P  = np.log(P)
H_self = np.sum(P * log_P, axis=1)        # (N,)
cross  = P @ log_P.T                       # (N, N)
D      = H_self[:, np.newaxis] - cross     # KL(i||j)
np.fill_diagonal(D, 0.0)
D      = np.maximum(D, 0.0)
D_sym  = 0.5 * (D + D.T)                  # symmetric
np.fill_diagonal(D_sym, 0.0)
del P, log_P, H_self, cross, D
gc.collect()
log(f"KL matrix done in {time.time()-t0:.2f}s")


# =========================================================
# SILHOUETTE SCORES
# =========================================================

overall_sil = silhouette_score(D_sym, labels, metric="precomputed")
sample_sils = silhouette_samples(D_sym, labels, metric="precomputed")

# Named-species-only silhouette (excludes heterogeneous Brucella_sp group)
named_mask    = labels != 7
named_labels  = labels[named_mask]
named_D       = D_sym[np.ix_(named_mask, named_mask)]
valid_classes = {c for c, n in Counter(named_labels.tolist()).items() if n >= 2}
valid_mask    = np.array([l in valid_classes for l in named_labels])
named_sil     = None
if valid_mask.sum() >= 2 and len(set(named_labels[valid_mask])) >= 2:
    named_sil = silhouette_score(
        named_D[np.ix_(valid_mask, valid_mask)],
        named_labels[valid_mask],
        metric="precomputed"
    )

log(f"\nOverall Silhouette (all 29 samples): {overall_sil:.4f}")
log(f"Silhouette (named species only):     "
    + (f"{named_sil:.4f}" if named_sil else "N/A"))

log(f"\n{'Sample':50s} {'Group':15s} {'Score':>10s}  Status")
log("-" * 90)
for name, lname, sc in zip(sample_names, label_names, sample_sils):
    status = classify_sample_status(sc)
    log(f"{name:50s} {lname:15s} {sc:10.4f}  {status}")


# =========================================================
# SAVE CSVs
# =========================================================

overall_file = f"{method_name}_k{k}_overall.csv"
with open(overall_file, "w") as f:
    f.write("method,k,alpha,vocabulary_size,zero_pct,"
            "overall_silhouette_score,named_silhouette_score\n")
    f.write(f"{method_name},{k},{alpha},{V},{zero_pct:.4f},"
            f"{overall_sil:.8f},"
            f"{named_sil:.8f if named_sil else float('nan')}\n")
log(f"\nSaved: {overall_file}")

per_sample_file = f"{method_name}_k{k}_per_sample.csv"
with open(per_sample_file, "w") as f:
    f.write("method,k,alpha,sample,class,sample_silhouette_score,status\n")
    for name, lname, sc in zip(sample_names, label_names, sample_sils):
        status = classify_sample_status(sc)
        f.write(f"{method_name},{k},{alpha},{name},{lname},{sc:.8f},{status}\n")
log(f"Saved: {per_sample_file}")


# =========================================================
# PER-SAMPLE BAR CHART
# =========================================================

x = np.arange(N)
plt.figure(figsize=(16, 5))
plt.bar(x, sample_sils)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, sample_names, rotation=90, fontsize=6)
plt.ylabel("Per-sample Silhouette Score")
plt.title(f"{method_name}  k={k}  alpha={alpha}  "
          f"overall={overall_sil:.4f}")
for idx, sc in enumerate(sample_sils):
    plt.text(idx, sc, f"{sc:.3f}", ha="center",
             va="bottom" if sc >= 0 else "top",
             fontsize=5, rotation=90)
plt.tight_layout()
fig_name = f"{method_name}_k{k}_per_sample.png"
plt.savefig(fig_name, dpi=150)
plt.close()
log(f"Saved: {fig_name}")

log(f"\nALL DONE for k={k}.")