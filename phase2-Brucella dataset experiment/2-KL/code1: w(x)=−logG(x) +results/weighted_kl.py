"""
SIWKL with w(x) = -log G(x) — low-memory Brucella version with metadata labels
================================================================================

This version:
  1. avoids the Out-Of-Memory problem by using sorted .npy arrays + mmap;
  2. reads sample labels from a metadata CSV;
  3. can reuse an already saved distance matrix to avoid recomputing everything.

Typical use when the matrix already exists:
    python3 weighted_kl_low_memory_with_labels.py \
      --k 15 --alpha 0.1 --beta 1.0 \
      --fasta_dir "../1-samples" \
      --out_dir "results_SIWKL_neg_log_G_lowmem" \
      --labels_csv "labels.csv" \
      --label_column group \
      --reuse_matrix

Full recomputation:
    python3 weighted_kl_low_memory_with_labels.py \
      --k 15 --alpha 0.1 --beta 1.0 \
      --fasta_dir "../1-samples" \
      --out_dir "results_SIWKL_neg_log_G_lowmem" \
      --labels_csv "labels.csv" \
      --label_column group \
      --chunk_size 100000
"""

import argparse
import gc
import glob
import os
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k",          type=int,   required=True)
    p.add_argument("--alpha",      type=float, required=True)
    p.add_argument("--beta",       type=float, required=True)
    p.add_argument("--fasta_dir",  type=str,   default="../1-samples")
    p.add_argument("--out_dir",    type=str,   default="results_SIWKL_neg_log_G_lowmem")
    p.add_argument("--chunk_size", type=int,   default=200_000)
    p.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Optional folder for intermediate .npy files. Default: system temporary folder.",
    )
    p.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV containing sample labels. Expected columns: sample and group/species.",
    )
    p.add_argument(
        "--sample_column",
        type=str,
        default="sample",
        help="Column name in labels_csv containing FASTA filenames.",
    )
    p.add_argument(
        "--label_column",
        type=str,
        default="group",
        help="Column name in labels_csv used as class label for silhouette. Recommended: group.",
    )
    p.add_argument(
        "--reuse_matrix",
        action="store_true",
        help="Load the already saved distance matrix from out_dir and only compute CSV/silhouette.",
    )
    return p.parse_args()


# =========================================================
# HELPERS
# =========================================================
SPECIES_KEYS = [
    "abortus", "melitensis", "suis", "canis", "ovis", "neotomae",
    "pinnipedialis", "ceti", "microti", "inopinata", "vulpis", "papionis"
]


def species_from_filename(fn):
    nl = fn.lower()
    for sp in SPECIES_KEYS:
        if sp in nl:
            return sp
    return "Brucella_sp"


def status_label(s):
    if pd.isna(s):
        return "not_computed"
    if s > 0.05:
        return "clear"
    if s < 0:
        return "possibly_misplaced"
    return "weak/ambiguous"


def matrix_filename(k, alpha, beta):
    return f"SIWKL_neg_log_G_matrix_k{k}_alpha{alpha}_beta{beta}.txt"


def csv_filename(k, alpha, beta):
    return f"SIWKL_neg_log_G_per_sample_k{k}_alpha{alpha}_beta{beta}.csv"


def load_metadata_labels(labels_csv, sample_names, sample_column="sample", label_column="group"):
    """Return labels aligned to sample_names, plus metadata rows aligned to sample_names."""
    if labels_csv is None:
        print("  No labels_csv provided. Falling back to filename-based labels.", flush=True)
        labels = [species_from_filename(n) for n in sample_names]
        meta = pd.DataFrame({"sample": sample_names, "label_source": "filename_fallback"})
        return labels, meta

    label_df = pd.read_csv(labels_csv)

    if sample_column not in label_df.columns:
        raise ValueError(
            f"labels_csv does not contain sample column '{sample_column}'. "
            f"Available columns: {list(label_df.columns)}"
        )
    if label_column not in label_df.columns:
        raise ValueError(
            f"labels_csv does not contain label column '{label_column}'. "
            f"Available columns: {list(label_df.columns)}"
        )

    # Normalize to strings and remove accidental spaces.
    label_df[sample_column] = label_df[sample_column].astype(str).str.strip()
    label_df[label_column] = label_df[label_column].astype(str).str.strip()

    if label_df[sample_column].duplicated().any():
        dupes = label_df.loc[label_df[sample_column].duplicated(), sample_column].tolist()
        raise ValueError(f"Duplicate sample names in labels_csv: {dupes[:5]}")

    label_map = dict(zip(label_df[sample_column], label_df[label_column]))
    missing = [s for s in sample_names if s not in label_map]
    if missing:
        raise ValueError(
            "Some FASTA files have no label in labels_csv. First missing examples: "
            + ", ".join(missing[:5])
        )

    extra = [s for s in label_df[sample_column].tolist() if s not in set(sample_names)]
    if extra:
        print(
            f"  Warning: {len(extra)} rows in labels_csv do not match FASTA files. "
            "They will be ignored.",
            flush=True,
        )

    # Align metadata to the same order as the distance matrix/sample_names.
    aligned = pd.DataFrame({sample_column: sample_names})
    meta = aligned.merge(label_df, on=sample_column, how="left")

    labels = meta[label_column].astype(str).tolist()

    print(f"  Loaded labels from: {labels_csv}", flush=True)
    print(f"  Label column used for silhouette: {label_column}", flush=True)
    print("  Label counts:", flush=True)
    for lab, cnt in pd.Series(labels).value_counts().items():
        print(f"    {lab}: {cnt}", flush=True)

    return labels, meta


def stream_kmers(path, k):
    """Yield k-mers from a FASTA file without loading the full sequence."""
    buf = []
    buf_len = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue

            line = line.upper()
            buf.append(line)
            buf_len += len(line)

            # Process in blocks. Keep k-1 bases as overlap for the next block.
            if buf_len > 5_000_000:
                seq = "".join(buf)
                limit = len(seq) - k + 1
                for i in range(max(0, limit)):
                    km = seq[i:i + k]
                    if "N" not in km:
                        yield km

                overlap = seq[-(k - 1):] if k > 1 else ""
                buf = [overlap] if overlap else []
                buf_len = len(overlap)

    if buf:
        seq = "".join(buf)
        limit = len(seq) - k + 1
        for i in range(max(0, limit)):
            km = seq[i:i + k]
            if "N" not in km:
                yield km


def count_kmers_streaming(path, k):
    """Count k-mers in one FASTA file. Only one genome is held at a time."""
    counts = Counter()
    for km in stream_kmers(path, k):
        counts[km] += 1
    return counts


def strings_to_fixed_bytes_array(keys, k):
    """Convert sorted Python string k-mers to a compact fixed-width byte array."""
    return np.fromiter(
        (km.encode("ascii") for km in keys),
        dtype=f"S{k}",
        count=len(keys),
    )


def save_sample_arrays(counts, k, idx, work_dir):
    """
    Save one sample as sorted fixed-byte keys and int32 counts.
    The sorted key array allows vectorized searchsorted in Pass 2.
    """
    keys_sorted = sorted(counts)
    keys_arr = strings_to_fixed_bytes_array(keys_sorted, k)
    vals_arr = np.fromiter(
        (counts[km] for km in keys_sorted),
        dtype=np.int32,
        count=len(keys_sorted),
    )

    keys_path = os.path.join(work_dir, f"sample_{idx:03d}_keys.npy")
    vals_path = os.path.join(work_dir, f"sample_{idx:03d}_vals.npy")
    np.save(keys_path, keys_arr)
    np.save(vals_path, vals_arr)

    del keys_sorted, keys_arr, vals_arr
    gc.collect()
    return keys_path, vals_path


# =========================================================
# PASS 1: count per genome and build global counts
# =========================================================
def pass1(files, k, work_dir):
    """
    For each genome:
      - count k-mers by streaming the FASTA file;
      - save sample counts as sorted .npy arrays;
      - update only one global Counter.

    Important memory fix:
      This avoids storing all per-sample Python dictionaries at once.
    """
    global_counts = Counter()
    raw_totals = []
    sample_array_paths = []

    for idx, fpath in enumerate(files):
        bname = os.path.basename(fpath)
        print(f"  [{idx + 1}/{len(files)}] counting {bname} ...", flush=True)

        counts = count_kmers_streaming(fpath, k)
        raw_totals.append(sum(counts.values()))
        global_counts.update(counts)

        sample_array_paths.append(save_sample_arrays(counts, k, idx, work_dir))

        del counts
        gc.collect()

    print("  Building compact global vocabulary arrays ...", flush=True)
    vocab_sorted = sorted(global_counts)
    V = len(vocab_sorted)

    vocab_keys = strings_to_fixed_bytes_array(vocab_sorted, k)
    global_vals = np.fromiter(
        (global_counts[km] for km in vocab_sorted),
        dtype=np.int64,
        count=V,
    )

    del vocab_sorted, global_counts
    gc.collect()

    return vocab_keys, global_vals, np.array(raw_totals, dtype=np.float64), sample_array_paths


# =========================================================
# PASS 2: chunked matrix computation with memory-mapped arrays
# =========================================================
def fill_probability_row(row, chunk_keys, sample_keys, sample_vals, alpha, smoothed_total):
    """
    Fill one row of P_chunk for one sample and one vocabulary chunk.
    Uses binary search over sorted sample_keys. No Python dict is created.
    """
    C = len(chunk_keys)
    row.fill(alpha)

    if len(sample_keys) > 0:
        pos = np.searchsorted(sample_keys, chunk_keys, side="left")
        valid = pos < len(sample_keys)

        found = np.zeros(C, dtype=bool)
        if np.any(valid):
            valid_pos = pos[valid]
            found[valid] = sample_keys[valid_pos] == chunk_keys[valid]

        if np.any(found):
            row[found] += sample_vals[pos[found]].astype(np.float64, copy=False)

    row /= smoothed_total


def compute_matrix_chunked(sample_array_paths, vocab_keys, global_vals,
                           alpha, beta, raw_totals, chunk_size):
    n = len(sample_array_paths)
    V = len(vocab_keys)

    smoothed_totals = raw_totals + alpha * V
    global_raw_total = float(global_vals.sum(dtype=np.int64))
    smoothed_global_total = global_raw_total + beta * V

    # W_max: rarest possible k-mer has raw global count 0, so G = beta / total.
    W_max = -np.log(beta / smoothed_global_total)
    if W_max <= 0:
        W_max = 1.0

    n_chunks = (V + chunk_size - 1) // chunk_size
    print(f"  V={V:,}  W_max={W_max:.4f}  chunks={n_chunks}", flush=True)
    print("  Opening per-sample arrays using mmap, not Python dictionaries ...", flush=True)

    sample_maps = []
    for keys_path, vals_path in sample_array_paths:
        sample_maps.append((
            np.load(keys_path, mmap_mode="r"),
            np.load(vals_path, mmap_mode="r"),
        ))

    M = np.zeros((n, n), dtype=np.float64)

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, V)
        chunk_keys = vocab_keys[c_start:c_end]
        C = len(chunk_keys)

        # --- P chunk: n × C ---
        P_chunk = np.empty((n, C), dtype=np.float64)
        for i, (sample_keys, sample_vals) in enumerate(sample_maps):
            fill_probability_row(
                P_chunk[i],
                chunk_keys,
                sample_keys,
                sample_vals,
                alpha,
                smoothed_totals[i],
            )

        # --- G and W chunks ---
        G_chunk = global_vals[c_start:c_end].astype(np.float64, copy=True)
        G_chunk += beta
        G_chunk /= smoothed_global_total
        W_chunk = -np.log(G_chunk) / W_max

        # --- Accumulate pairwise symmetric weighted KL contributions ---
        logP = np.log(P_chunk)
        for i in range(n):
            Pi = P_chunk[i]
            logPi = logP[i]
            for j in range(i + 1, n):
                diff = logPi - logP[j]
                contrib = 0.5 * np.dot(W_chunk, Pi * diff - P_chunk[j] * diff)
                M[i, j] += contrib
                M[j, i] += contrib

        del P_chunk, G_chunk, W_chunk, logP
        gc.collect()

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci + 1}/{n_chunks} done", flush=True)

    del sample_maps
    gc.collect()

    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 0.0)
    M[M < 0] = 0.0
    return M


# =========================================================
# RESULTS: silhouette + CSV
# =========================================================
def compute_silhouette_scores(M, class_labels):
    """
    Compute silhouette only on non-singleton classes.
    Singletons receive NaN because silhouette is undefined for singleton classes.
    """
    class_labels = np.array(class_labels, dtype=object)
    unique, cnts = np.unique(class_labels, return_counts=True)
    valid_classes = unique[cnts >= 2]
    valid_mask = np.isin(class_labels, valid_classes)
    per_sample_scores = np.full(len(class_labels), np.nan, dtype=np.float64)

    sub_indices = np.where(valid_mask)[0]
    if len(sub_indices) < 3:
        print("Silhouette skipped: not enough non-singleton labeled samples.", flush=True)
        return per_sample_scores, np.nan

    sub_M = M[np.ix_(valid_mask, valid_mask)]
    sub_labels = class_labels[valid_mask]
    n_sub_labels = len(np.unique(sub_labels))

    if not (2 <= n_sub_labels <= (len(sub_labels) - 1)):
        print(
            "Silhouette skipped: labels contain "
            f"{n_sub_labels} class(es) after singleton filtering. "
            "Silhouette needs at least 2 classes.",
            flush=True,
        )
        return per_sample_scores, np.nan

    sub_scores = silhouette_samples(sub_M, sub_labels, metric="precomputed")
    for ii, orig in enumerate(sub_indices):
        per_sample_scores[orig] = sub_scores[ii]

    gsil = silhouette_score(sub_M, sub_labels, metric="precomputed")
    print(f"Global silhouette (non-singleton classes): {gsil:.6f}", flush=True)
    print(
        "  Note: singleton classes are excluded from silhouette and written as NaN.",
        flush=True,
    )
    return per_sample_scores, gsil


def save_outputs(M, sample_names, labels, metadata, args):
    os.makedirs(args.out_dir, exist_ok=True)

    matrix_path = os.path.join(args.out_dir, matrix_filename(args.k, args.alpha, args.beta))
    if not args.reuse_matrix:
        np.savetxt(matrix_path, M, fmt="%.8e")
        print(f"\nSaved distance matrix: {matrix_path}", flush=True)
    else:
        print(f"\nUsing existing distance matrix: {matrix_path}", flush=True)

    per_sample_scores, gsil = compute_silhouette_scores(M, labels)

    # Build output rows and keep useful metadata columns from labels.csv.
    out = pd.DataFrame({
        "method": "SIWKL_neg_log_G",
        "k": args.k,
        "alpha": args.alpha,
        "beta": args.beta,
        "sample": sample_names,
        "class": labels,
        "sample_silhouette_score": [
            np.nan if pd.isna(x) else round(float(x), 8) for x in per_sample_scores
        ],
        "status": [status_label(x) for x in per_sample_scores],
        "global_silhouette_non_singleton": np.nan if pd.isna(gsil) else round(float(gsil), 8),
    })

    # Merge metadata columns, excluding duplicate sample/class columns.
    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        # Rename sample column to sample if needed.
        if args.sample_column in meta.columns and args.sample_column != "sample":
            meta = meta.rename(columns={args.sample_column: "sample"})
        drop_cols = [c for c in meta.columns if c in out.columns and c != "sample"]
        meta = meta.drop(columns=drop_cols, errors="ignore")
        out = out.merge(meta, on="sample", how="left")

    csv_path = os.path.join(args.out_dir, csv_filename(args.k, args.alpha, args.beta))
    out.to_csv(csv_path, index=False)
    print(f"Saved per-sample results CSV: {csv_path}\n", flush=True)
    print(out.to_string(index=False), flush=True)


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.fasta_dir, "*.fa")))
    if not files:
        raise RuntimeError(f"No .fa files found in: {args.fasta_dir}")

    sample_names = [os.path.basename(f) for f in files]
    class_labels, metadata = load_metadata_labels(
        args.labels_csv,
        sample_names,
        sample_column=args.sample_column,
        label_column=args.label_column,
    )

    print(f"\n{'=' * 60}")
    print("  SIWKL  w(x) = -log G(x)  [LOW-MEMORY + METADATA LABELS]")
    print(f"  k={args.k}  alpha={args.alpha}  beta={args.beta}")
    print(f"{'=' * 60}")
    print(f"  {len(files)} genomes found")

    matrix_path = os.path.join(args.out_dir, matrix_filename(args.k, args.alpha, args.beta))

    if args.reuse_matrix:
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"--reuse_matrix was requested, but this matrix file does not exist: {matrix_path}"
            )
        print(f"\n[Reuse] Loading existing matrix: {matrix_path}", flush=True)
        M = np.loadtxt(matrix_path)
        if M.shape != (len(sample_names), len(sample_names)):
            raise ValueError(
                f"Matrix shape {M.shape} does not match number of FASTA samples {len(sample_names)}. "
                "Check that fasta_dir is the same folder/order used when computing the matrix."
            )
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
        save_outputs(M, sample_names, class_labels, metadata, args)
        return

    # Use either the user-provided temporary folder or a safe system temp folder.
    temp_context = (
        tempfile.TemporaryDirectory(dir=args.tmp_dir)
        if args.tmp_dir is not None else tempfile.TemporaryDirectory()
    )

    with temp_context as work_dir:
        print(f"  Temporary work directory: {work_dir}", flush=True)

        # --- Pass 1 ---
        print("\n[Pass 1] Streaming k-mer counts per genome...")
        vocab_keys, global_vals, raw_totals, sample_array_paths = pass1(files, args.k, work_dir)
        print(f"  Union vocabulary size: {len(vocab_keys):,}")

        # --- Pass 2 ---
        print("\n[Pass 2] Computing chunked SIWKL matrix...")
        M = compute_matrix_chunked(
            sample_array_paths,
            vocab_keys,
            global_vals,
            args.alpha,
            args.beta,
            raw_totals,
            args.chunk_size,
        )

        del vocab_keys, global_vals, raw_totals, sample_array_paths
        gc.collect()

    save_outputs(M, sample_names, class_labels, metadata, args)


if __name__ == "__main__":
    main()