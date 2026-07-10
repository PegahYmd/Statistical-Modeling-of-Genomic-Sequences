"""
Cosine Similarity (k-mer based) — Brucella dataset
====================================================
Memory strategy:
  Pass 1 – stream each genome, save k-mer counts to disk (.npz), build
            union vocab. Never hold more than 1 genome in RAM.
  Pass 2 – load one genome at a time, compute frequency vector in chunks,
            accumulate dot-products and norms for cosine distance matrix.

Output CSV format matches baseline_KL_w1 / SIWKL_neg_log_G files:
    method, k, sample, class, sample_silhouette_score, status

Usage:
    python3 cosine_brucella.py --k 15
    python3 cosine_brucella.py --k 31
    python3 cosine_brucella.py --k 15 --chunk_size 100000
"""

import argparse, os, glob, gc, tempfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k",          type=int, required=True,  help="k-mer length")
    p.add_argument("--fasta_dir",  type=str, default="./1-samples")
    p.add_argument("--out_dir",    type=str, default="results_cosine")
    p.add_argument("--chunk_size", type=int, default=200_000,
                   help="Vocab columns per chunk (lower = less RAM)")
    return p.parse_args()


# =========================================================
# SPECIES LABELS  (Brucella)
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


# =========================================================
# STREAMING K-MER COUNTER
# =========================================================
def count_kmers_streaming(path, k):
    """Count k-mers in a FASTA file without loading the full sequence into RAM."""
    counts = Counter()
    buf    = []
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
                    km = seq[i:i+k]
                    if "N" not in km:
                        counts[km] += 1
                buf     = [seq[-(k-1):]]   # keep overlap
                buf_len = k - 1

    if buf:
        seq = "".join(buf)
        for i in range(len(seq) - k + 1):
            km = seq[i:i+k]
            if "N" not in km:
                counts[km] += 1

    return counts


# =========================================================
# PASS 1 — count & save to disk
# =========================================================
def pass1(files, k, tmp_dir):
    union_vocab = set()
    raw_totals  = []
    npz_paths   = []

    for idx, fpath in enumerate(files):
        print(f"  [{idx+1}/{len(files)}] {os.path.basename(fpath)} ...", flush=True)

        counts = count_kmers_streaming(fpath, k)
        raw_totals.append(sum(counts.values()))
        union_vocab.update(counts.keys())

        keys_arr = np.array(list(counts.keys()),   dtype=object)
        vals_arr = np.array(list(counts.values()), dtype=np.int32)

        npz_path = os.path.join(tmp_dir, f"sample_{idx:03d}.npz")
        np.savez_compressed(npz_path, keys=keys_arr, vals=vals_arr)
        npz_paths.append(npz_path)

        del counts, keys_arr, vals_arr
        gc.collect()

    vocab = sorted(union_vocab)
    del union_vocab
    gc.collect()

    return vocab, np.array(raw_totals, dtype=np.float64), npz_paths


def load_counts(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return dict(zip(data["keys"], data["vals"].tolist()))


# =========================================================
# PASS 2 — chunked cosine distance matrix
# =========================================================
def compute_cosine_chunked(npz_paths, vocab, raw_totals, chunk_size):
    """
    Cosine similarity = (P_i · P_j) / (||P_i|| ||P_j||)
    where P_i is the normalised k-mer frequency vector.

    We compute dot-products and squared norms in vocabulary chunks
    without ever materialising the full (n × V) matrix.
    """
    n  = len(npz_paths)
    V  = len(vocab)
    n_chunks = (V + chunk_size - 1) // chunk_size

    print(f"  V={V:,}  n_chunks={n_chunks}", flush=True)

    # Accumulators
    dot_products = np.zeros((n, n), dtype=np.float64)
    sq_norms     = np.zeros(n,      dtype=np.float64)

    print("  Loading per-sample count dicts ...", flush=True)
    sample_dicts = [load_counts(p) for p in npz_paths]

    for ci in range(n_chunks):
        c_start     = ci * chunk_size
        c_end       = min(c_start + chunk_size, V)
        chunk_vocab = vocab[c_start:c_end]
        C           = len(chunk_vocab)

        # Build frequency chunk (n × C)
        F_chunk = np.empty((n, C), dtype=np.float64)
        for i, sd in enumerate(sample_dicts):
            for j, km in enumerate(chunk_vocab):
                F_chunk[i, j] = sd.get(km, 0)
        # Normalise to frequencies
        F_chunk /= raw_totals[:, np.newaxis]

        # Accumulate dot products
        dot_products += F_chunk @ F_chunk.T   # (n × C) @ (C × n)

        # Accumulate squared norms
        sq_norms += np.sum(F_chunk ** 2, axis=1)

        del F_chunk
        gc.collect()

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci+1}/{n_chunks} done", flush=True)

    del sample_dicts
    gc.collect()

    # Cosine similarity
    norms = np.sqrt(sq_norms)                           # (n,)
    outer_norms = np.outer(norms, norms)                # (n, n)
    outer_norms[outer_norms == 0] = 1e-12               # avoid /0

    sim_matrix = dot_products / outer_norms
    np.fill_diagonal(sim_matrix, 1.0)                   # numerical safety
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, None)

    return sim_matrix, dist_matrix


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

    sample_names   = [os.path.basename(f) for f in files]
    species_labels = [species_from_filename(n) for n in sample_names]

    print(f"\n{'='*60}")
    print(f"  Cosine Similarity (k-mer)")
    print(f"  k={k}")
    print(f"{'='*60}")
    print(f"  {len(files)} genomes")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # --- Pass 1 ---
        print("\n[Pass 1] Streaming k-mer counts per genome...")
        vocab, raw_totals, npz_paths = pass1(files, k, tmp_dir)
        print(f"  Union vocabulary: {len(vocab):,}")

        # --- Pass 2 ---
        print("\n[Pass 2] Computing chunked cosine matrix...")
        sim_matrix, dist_matrix = compute_cosine_chunked(
            npz_paths, vocab, raw_totals, args.chunk_size
        )

        del vocab, raw_totals
        gc.collect()

    # --- Save matrices ---
    sim_df  = pd.DataFrame(sim_matrix,  index=sample_names, columns=sample_names)
    dist_df = pd.DataFrame(dist_matrix, index=sample_names, columns=sample_names)
    sim_df.to_csv( os.path.join(args.out_dir, f"cosine_similarity_matrix_k{k}.csv"))
    dist_df.to_csv(os.path.join(args.out_dir, f"cosine_distance_matrix_k{k}.csv"))
    print(f"\nSaved similarity and distance matrices (k={k})")

    # --- Silhouette (same logic as SIWKL scripts) ---
    unique, cnts = np.unique(species_labels, return_counts=True)
    valid_mask   = np.isin(species_labels, unique[cnts >= 2])
    per_sample_scores = np.zeros(len(sample_names))

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

    # --- Output CSV (matches baseline_KL_w1 / SIWKL format) ---
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