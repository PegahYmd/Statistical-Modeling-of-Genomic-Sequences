"""
Cosine Similarity (k-mer based) — Brucella dataset
====================================================
Memory strategy:
  - Encode k-mers as 64-bit integers (no string storage at all)
  - Store each genome as a scipy sparse vector on disk
  - Compute cosine dot-products genome-pair by genome-pair (n*(n-1)/2 pairs)
    loading only 2 sparse vectors at a time
  - For k=15: 4^15 = ~1B possible k-mers but only ~500K observed per genome
    → sparse vectors are tiny

Usage:
    python3 cosine_brucella.py --k 15
    python3 cosine_brucella.py --k 31
"""

import argparse, os, glob, gc, tempfile
from collections import Counter

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k",         type=int, required=True)
    p.add_argument("--fasta_dir", type=str, default="./1-samples")
    p.add_argument("--out_dir",   type=str, default="results_cosine")
    return p.parse_args()


# =========================================================
# SPECIES LABELS
# =========================================================
SPECIES_KEYS = [
    "abortus","melitensis","suis","canis","ovis","neotomae",
    "pinnipedialis","ceti","microti","inopinata","vulpis","papionis"
]

def species_from_filename(fn):
    nl = fn.lower()
    for sp in SPECIES_KEYS:
        if sp in nl:
            return sp
    return "Brucella_sp"

def status_label(s):
    if s > 0.05: return "clear"
    if s < 0:    return "possibly_misplaced"
    return "weak/ambiguous"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def kmer_to_int(km):
    """Encode k-mer as a base-4 integer. Returns -1 if invalid base."""
    val = 0
    for c in km:
        b = BASE_MAP.get(c, -1)
        if b == -1:
            return -1
        val = val * 4 + b
    return val


# =========================================================
# STREAM AND COUNT — returns (int_indices, counts) arrays
# =========================================================
def count_kmers_as_ints(path, k):
    """
    Stream FASTA, count k-mers encoded as integers.
    Returns two int64 arrays: unique indices and their counts.
    Peak RAM: one Counter of integers (8 bytes/key vs 60 bytes for strings).
    """
    counts  = Counter()
    buf     = []
    buf_len = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            buf.append(line.upper())
            buf_len += len(line)
            if buf_len >= 5_000_000:
                seq = "".join(buf)
                for i in range(len(seq) - k + 1):
                    km  = seq[i:i+k]
                    idx = kmer_to_int(km)
                    if idx >= 0:
                        counts[idx] += 1
                buf     = [seq[-(k-1):]]
                buf_len = k - 1

    if buf:
        seq = "".join(buf)
        for i in range(len(seq) - k + 1):
            km  = seq[i:i+k]
            idx = kmer_to_int(km)
            if idx >= 0:
                counts[idx] += 1

    indices = np.array(list(counts.keys()),   dtype=np.int64)
    values  = np.array(list(counts.values()), dtype=np.float32)
    del counts
    return indices, values


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    k    = args.k

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.fasta_dir, "*.fa")))
    if not files:
        raise RuntimeError(f"No .fa files found in: {args.fasta_dir}")

    n              = len(files)
    sample_names   = [os.path.basename(f) for f in files]
    species_labels = [species_from_filename(nm) for nm in sample_names]

    print(f"\n{'='*60}")
    print(f"  Cosine Similarity (k-mer)  k={k}")
    print(f"{'='*60}")
    print(f"  {n} genomes")

    # Max possible index for this k
    max_idx = 4 ** k

    with tempfile.TemporaryDirectory() as tmp_dir:

        # -------------------------------------------------------
        # Pass 1: count each genome, save sparse vector to disk
        # -------------------------------------------------------
        print("\n[Pass 1] Counting k-mers (integer-encoded)...")
        npz_paths  = []
        raw_totals = np.zeros(n, dtype=np.float64)

        for i, fpath in enumerate(files):
            print(f"  [{i+1}/{n}] {os.path.basename(fpath)}", flush=True)
            indices, values = count_kmers_as_ints(fpath, k)
            raw_totals[i]   = values.sum()
            # Normalise to frequencies immediately — store floats
            freq = values / raw_totals[i]
            npz_path = os.path.join(tmp_dir, f"s{i:03d}.npz")
            np.savez_compressed(npz_path, idx=indices, freq=freq)
            npz_paths.append(npz_path)
            del indices, values, freq
            gc.collect()

        print(f"  Done. Totals range: {raw_totals.min():.0f} – {raw_totals.max():.0f} k-mers")

        # -------------------------------------------------------
        # Pass 2: pairwise cosine — load 2 vectors at a time
        # -------------------------------------------------------
        print("\n[Pass 2] Computing pairwise cosine (sparse, 2 vectors at a time)...")

        def load_sparse(npz_path):
            d = np.load(npz_path)
            return d["idx"].astype(np.int64), d["freq"].astype(np.float64)

        # Pre-compute norms (load each genome once)
        norms = np.zeros(n, dtype=np.float64)
        for i in range(n):
            _, freq = load_sparse(npz_paths[i])
            norms[i] = np.sqrt(np.dot(freq, freq))
            del freq

        sim_matrix = np.eye(n, dtype=np.float64)
        total_pairs = n * (n - 1) // 2
        done = 0

        for i in range(n):
            idx_i, freq_i = load_sparse(npz_paths[i])
            # Build dict for fast lookup
            dict_i = dict(zip(idx_i, freq_i))

            for j in range(i + 1, n):
                idx_j, freq_j = load_sparse(npz_paths[j])

                # Dot product over shared k-mers only
                shared = np.intersect1d(idx_i, idx_j, assume_unique=True)
                if len(shared) > 0:
                    vals_i = np.array([dict_i[x] for x in shared])
                    vals_j = freq_j[np.searchsorted(idx_j, shared)]
                    dot    = np.dot(vals_i, vals_j)
                else:
                    dot = 0.0

                denom = norms[i] * norms[j]
                sim   = dot / denom if denom > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

                del idx_j, freq_j, shared
                done += 1

            del idx_i, freq_i, dict_i
            gc.collect()

            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  row {i+1}/{n}  ({done}/{total_pairs} pairs)", flush=True)

    # -------------------------------------------------------
    # Finalise
    # -------------------------------------------------------
    np.fill_diagonal(sim_matrix, 1.0)
    sim_matrix  = np.clip(sim_matrix, -1.0, 1.0)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, None)
    np.fill_diagonal(dist_matrix, 0.0)

    # Save matrices
    pd.DataFrame(sim_matrix,  index=sample_names, columns=sample_names).to_csv(
        os.path.join(args.out_dir, f"cosine_similarity_matrix_k{k}.csv"))
    pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names).to_csv(
        os.path.join(args.out_dir, f"cosine_distance_matrix_k{k}.csv"))
    print(f"\nSaved similarity and distance matrices")

    # Silhouette
    unique, cnts = np.unique(species_labels, return_counts=True)
    valid_mask   = np.isin(species_labels, unique[cnts >= 2])
    per_sample_scores = np.zeros(n)

    if valid_mask.sum() >= 2:
        sub_M      = dist_matrix[np.ix_(valid_mask, valid_mask)]
        sub_labels = np.array(species_labels)[valid_mask]
        sub_scores = silhouette_samples(sub_M, sub_labels, metric="precomputed")
        for ii, orig in enumerate(np.where(valid_mask)[0]):
            per_sample_scores[orig] = sub_scores[ii]
        gsil = silhouette_score(sub_M, sub_labels, metric="precomputed")
        print(f"Global silhouette (non-singleton species): {gsil:.6f}")
    else:
        print("Not enough non-singleton species for global silhouette.")

    # Output CSV
    rows = []
    for sname, sp, sc in zip(sample_names, species_labels, per_sample_scores):
        rows.append({
            "method":                  "cosine",
            "k":                       k,
            "sample":                  sname,
            "class":                   sp,
            "sample_silhouette_score": round(float(sc), 8),
            "status":                  status_label(sc)
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, f"cosine_per_sample_silhouette_k{k}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()