"""
Jensen-Shannon Distance on k-mer distributions — low-memory version with metadata labels
=====================================================================================

This version is adapted from the original jsd.py script, but avoids building the full
N x V count/probability matrix in RAM. It:
  1. streams FASTA files instead of reading all sequences into memory;
  2. saves per-sample k-mer counts as sorted .npy arrays;
  3. computes JSD/JS distance in vocabulary chunks;
  4. reads labels from labels.csv instead of inferring labels from filenames;
  5. optionally reuses an already saved JS distance matrix.

Typical single run:
    python3 jsd_low_memory_with_labels.py \
      --k 15 \
      --fasta_dir "../1-samples" \
      --out_dir "JSD_results_lowmem" \
      --labels_csv "labels.csv" \
      --label_column group \
      --chunk_size 100000

Reuse an already saved matrix and only recompute silhouette/CSV:
    python3 jsd_low_memory_with_labels.py \
      --k 15 \
      --fasta_dir "../1-samples" \
      --out_dir "JSD_results_lowmem" \
      --labels_csv "labels.csv" \
      --label_column group \
      --reuse_matrix

Grid run:
    python3 jsd_low_memory_with_labels.py \
      --run_grid \
      --k_values 15 21 31 \
      --fasta_dir "../1-samples" \
      --out_dir "JSD_results_lowmem" \
      --labels_csv "labels.csv" \
      --label_column group \
      --chunk_size 100000
"""

import argparse
import gc
import glob
import math
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

    # Single-run parameters
    p.add_argument("--k", type=int, default=15)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Optional additive smoothing. Use 0.0 for standard JSD, matching the original script.",
    )
    p.add_argument("--log_base", type=float, default=2.0)

    # Grid-run parameters
    p.add_argument("--run_grid", action="store_true")
    p.add_argument("--k_values", type=int, nargs="+", default=[15, 21, 31])

    p.add_argument("--fasta_dir", type=str, default="../1-samples")
    p.add_argument("--out_dir", type=str, default="JSD_results_lowmem")
    p.add_argument("--chunk_size", type=int, default=100_000)
    p.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Optional parent folder for intermediate .npy files. Default: system temporary folder.",
    )

    # Labels
    p.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV containing sample labels. Recommended columns: sample, species, group, group_id.",
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
        help="Load the already saved JS distance matrix from out_dir and only compute CSV/silhouette. Only for single-run mode.",
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


def status_label(score):
    if pd.isna(score):
        return "not_computed"
    if score > 0.05:
        return "clear"
    if score < 0:
        return "possibly_misplaced"
    return "weak/ambiguous"


def float_tag(x):
    """Make a float filename-safe but still readable."""
    return str(x).replace(".", "p").replace("-", "m")


def jsd_matrix_filename(k, alpha, log_base):
    return f"JSD_matrix_k{k}_alpha{alpha}_logbase{log_base}.txt"


def js_distance_matrix_filename(k, alpha, log_base):
    return f"JS_distance_matrix_k{k}_alpha{alpha}_logbase{log_base}.txt"


def per_sample_filename(k, alpha, log_base):
    return f"JS_distance_per_sample_silhouette_k{k}_alpha{alpha}_logbase{log_base}.csv"


def overall_filename():
    return "JS_distance_global_results.csv"


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
            f"  Warning: {len(extra)} rows in labels_csv do not match FASTA files. They will be ignored.",
            flush=True,
        )

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
    """Yield valid A/C/G/T k-mers from a FASTA file without loading the full sequence."""
    buf = []
    buf_len = 0
    valid = set("ACGT")

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
                    if set(km).issubset(valid):
                        yield km

                overlap = seq[-(k - 1):] if k > 1 else ""
                buf = [overlap] if overlap else []
                buf_len = len(overlap)

    if buf:
        seq = "".join(buf)
        limit = len(seq) - k + 1
        for i in range(max(0, limit)):
            km = seq[i:i + k]
            if set(km).issubset(valid):
                yield km


def count_kmers_streaming(path, k):
    """Count k-mers in one FASTA file. Only one genome is held at a time."""
    counts = Counter()
    for km in stream_kmers(path, k):
        counts[km] += 1
    return counts


def strings_to_fixed_bytes_array(keys, k):
    """Convert sorted Python string k-mers to a compact fixed-width byte array."""
    return np.fromiter((km.encode("ascii") for km in keys), dtype=f"S{k}", count=len(keys))


def save_sample_arrays(counts, k, idx, work_dir):
    """
    Save one sample as sorted fixed-byte keys and int32 counts.
    The sorted key array allows vectorized searchsorted in Pass 2.
    """
    keys_sorted = sorted(counts)
    keys_arr = strings_to_fixed_bytes_array(keys_sorted, k)
    vals_arr = np.fromiter((counts[km] for km in keys_sorted), dtype=np.int32, count=len(keys_sorted))

    keys_path = os.path.join(work_dir, f"sample_{idx:03d}_keys.npy")
    vals_path = os.path.join(work_dir, f"sample_{idx:03d}_vals.npy")
    np.save(keys_path, keys_arr)
    np.save(vals_path, vals_arr)

    nnz = len(keys_sorted)
    del keys_sorted, keys_arr, vals_arr
    gc.collect()
    return keys_path, vals_path, nnz


# =========================================================
# PASS 1: count per genome and build global vocabulary
# =========================================================
def pass1(files, k, work_dir):
    """
    For each genome:
      - count k-mers by streaming the FASTA file;
      - save sample counts as sorted .npy arrays;
      - update one global Counter only to build the union vocabulary.

    This avoids storing all per-sample Python dictionaries at once.
    """
    global_counts = Counter()
    raw_totals = []
    sample_array_paths = []
    nnz_per_sample = []

    for idx, fpath in enumerate(files):
        bname = os.path.basename(fpath)
        print(f"  [{idx + 1}/{len(files)}] counting {bname} ...", flush=True)

        counts = count_kmers_streaming(fpath, k)
        raw_total = sum(counts.values())
        if raw_total == 0:
            raise ValueError(f"FASTA file produced zero valid k-mers for k={k}: {bname}")

        raw_totals.append(raw_total)
        global_counts.update(counts)

        keys_path, vals_path, nnz = save_sample_arrays(counts, k, idx, work_dir)
        sample_array_paths.append((keys_path, vals_path))
        nnz_per_sample.append(nnz)

        del counts
        gc.collect()

    print("  Building compact global vocabulary array ...", flush=True)
    vocab_sorted = sorted(global_counts)
    V = len(vocab_sorted)

    vocab_keys = strings_to_fixed_bytes_array(vocab_sorted, k)

    del vocab_sorted, global_counts
    gc.collect()

    raw_totals = np.array(raw_totals, dtype=np.float64)
    nnz_per_sample = np.array(nnz_per_sample, dtype=np.int64)
    zero_entries = len(files) * V - int(nnz_per_sample.sum())
    zero_percentage = 100.0 * zero_entries / (len(files) * V)

    return vocab_keys, raw_totals, sample_array_paths, nnz_per_sample, zero_percentage


# =========================================================
# PASS 2: chunked JSD/JS-distance computation
# =========================================================
def fill_probability_row(row, chunk_keys, sample_keys, sample_vals, alpha, denominator):
    """
    Fill one row of P_chunk for one sample and one vocabulary chunk.
    Uses binary search over sorted sample_keys. No Python dict is created.
    """
    C = len(chunk_keys)

    if alpha > 0:
        row.fill(alpha)
    else:
        row.fill(0.0)

    if len(sample_keys) > 0:
        pos = np.searchsorted(sample_keys, chunk_keys, side="left")
        valid = pos < len(sample_keys)

        found = np.zeros(C, dtype=bool)
        if np.any(valid):
            valid_pos = pos[valid]
            found[valid] = sample_keys[valid_pos] == chunk_keys[valid]

        if np.any(found):
            row[found] += sample_vals[pos[found]].astype(np.float64, copy=False)

    row /= denominator


def jsd_pair_from_chunks(Pi, Pj, inv_log_base):
    """
    Chunk contribution to JSD(Pi, Pj).
    Terms where P=0 contribute 0, which is important for alpha=0.
    """
    M = 0.5 * (Pi + Pj)

    mask_i = Pi > 0
    if np.any(mask_i):
        kl_i = np.sum(Pi[mask_i] * np.log(Pi[mask_i] / M[mask_i])) * inv_log_base
    else:
        kl_i = 0.0

    mask_j = Pj > 0
    if np.any(mask_j):
        kl_j = np.sum(Pj[mask_j] * np.log(Pj[mask_j] / M[mask_j])) * inv_log_base
    else:
        kl_j = 0.0

    return 0.5 * (kl_i + kl_j)


def compute_jsd_chunked(sample_array_paths, vocab_keys, raw_totals, alpha, log_base, chunk_size):
    n = len(sample_array_paths)
    V = len(vocab_keys)

    if log_base <= 0 or log_base == 1.0:
        raise ValueError("log_base must be positive and different from 1.")

    denominators = raw_totals + alpha * V
    if np.any(denominators <= 0):
        raise ValueError("Invalid probability denominators. Check FASTA files and alpha.")

    n_chunks = (V + chunk_size - 1) // chunk_size
    inv_log_base = 1.0 / np.log(log_base)

    print(f"  V={V:,}  chunks={n_chunks}  alpha={alpha}  log_base={log_base}", flush=True)
    if alpha == 0:
        print("  Probability mode: standard empirical k-mer distributions, no smoothing", flush=True)
    else:
        print(f"  Probability mode: additive smoothing with alpha={alpha}", flush=True)

    print("  Opening per-sample arrays using mmap, not Python dictionaries ...", flush=True)
    sample_maps = []
    for keys_path, vals_path in sample_array_paths:
        sample_maps.append((np.load(keys_path, mmap_mode="r"), np.load(vals_path, mmap_mode="r")))

    jsd_matrix = np.zeros((n, n), dtype=np.float64)

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, V)
        chunk_keys = vocab_keys[c_start:c_end]
        C = len(chunk_keys)

        P_chunk = np.empty((n, C), dtype=np.float64)
        for i, (sample_keys, sample_vals) in enumerate(sample_maps):
            fill_probability_row(
                P_chunk[i],
                chunk_keys,
                sample_keys,
                sample_vals,
                alpha,
                denominators[i],
            )

        for i in range(n):
            Pi = P_chunk[i]
            for j in range(i + 1, n):
                jsd_val = jsd_pair_from_chunks(Pi, P_chunk[j], inv_log_base)
                jsd_matrix[i, j] += jsd_val
                jsd_matrix[j, i] += jsd_val

        del P_chunk
        gc.collect()

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci + 1}/{n_chunks} done", flush=True)

    del sample_maps
    gc.collect()

    # Numerical cleanup
    jsd_matrix = 0.5 * (jsd_matrix + jsd_matrix.T)
    np.fill_diagonal(jsd_matrix, 0.0)
    jsd_matrix[jsd_matrix < 0] = 0.0

    # With log base 2, JSD should be <= 1. Small overshoots may happen from floating point.
    js_distance_matrix = np.sqrt(jsd_matrix)
    np.fill_diagonal(js_distance_matrix, 0.0)

    stats = {
        "min_jsd": float(np.min(jsd_matrix)),
        "max_jsd": float(np.max(jsd_matrix)),
        "min_js_distance": float(np.min(js_distance_matrix)),
        "max_js_distance": float(np.max(js_distance_matrix)),
    }
    return jsd_matrix, js_distance_matrix, stats


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
    return per_sample_scores, float(gsil)


def save_outputs(jsd_matrix, js_distance_matrix, sample_names, labels, metadata, args,
                 stats=None, zero_percentage=np.nan, vocabulary_size=np.nan,
                 matrix_already_saved=False):
    os.makedirs(args.out_dir, exist_ok=True)

    jsd_path = os.path.join(args.out_dir, jsd_matrix_filename(args.k, args.alpha, args.log_base))
    dist_path = os.path.join(args.out_dir, js_distance_matrix_filename(args.k, args.alpha, args.log_base))

    if not matrix_already_saved and not args.reuse_matrix:
        np.savetxt(jsd_path, jsd_matrix, fmt="%.8e")
        np.savetxt(dist_path, js_distance_matrix, fmt="%.8e")
        print(f"\nSaved JSD matrix: {jsd_path}", flush=True)
        print(f"Saved JS distance matrix: {dist_path}", flush=True)
    elif args.reuse_matrix:
        print(f"\nUsing existing JS distance matrix: {dist_path}", flush=True)

    per_sample_scores, gsil = compute_silhouette_scores(js_distance_matrix, labels)

    out = pd.DataFrame({
        "method": "Jensen_Shannon_distance",
        "k": args.k,
        "alpha": args.alpha,
        "log_base": args.log_base,
        "sample": sample_names,
        "class": labels,
        "sample_silhouette_score": [
            np.nan if pd.isna(x) else round(float(x), 8) for x in per_sample_scores
        ],
        "status": [status_label(x) for x in per_sample_scores],
        "global_silhouette_non_singleton": np.nan if pd.isna(gsil) else round(float(gsil), 8),
    })

    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        if args.sample_column in meta.columns and args.sample_column != "sample":
            meta = meta.rename(columns={args.sample_column: "sample"})
        drop_cols = [c for c in meta.columns if c in out.columns and c != "sample"]
        meta = meta.drop(columns=drop_cols, errors="ignore")
        out = out.merge(meta, on="sample", how="left")

    csv_path = os.path.join(args.out_dir, per_sample_filename(args.k, args.alpha, args.log_base))
    out.to_csv(csv_path, index=False)
    print(f"Saved per-sample results CSV: {csv_path}\n", flush=True)
    print(out.to_string(index=False), flush=True)

    overall_row = {
        "method": "Jensen_Shannon_distance",
        "k": args.k,
        "alpha": args.alpha,
        "log_base": args.log_base,
        "vocabulary_size": vocabulary_size,
        "zero_percentage_before_smoothing": zero_percentage,
        "overall_silhouette_score": gsil,
    }
    if stats:
        overall_row.update(stats)

    return overall_row


def run_single(args, files, sample_names, class_labels, metadata, precomputed=None):
    print(f"\n{'=' * 70}")
    print("  Jensen-Shannon Distance  [LOW-MEMORY + METADATA LABELS]")
    print(f"  k={args.k}  alpha={args.alpha}  log_base={args.log_base}")
    print(f"{'=' * 70}")
    print(f"  {len(files)} genomes found", flush=True)

    dist_path = os.path.join(args.out_dir, js_distance_matrix_filename(args.k, args.alpha, args.log_base))

    if args.reuse_matrix:
        if not os.path.exists(dist_path):
            raise FileNotFoundError(
                f"--reuse_matrix was requested, but this JS distance matrix file does not exist: {dist_path}\n"
                "Run the script once without --reuse_matrix first, or set --out_dir to the folder where the matrix was saved."
            )
        print(f"\n[Reuse] Loading existing JS distance matrix: {dist_path}", flush=True)
        js_distance_matrix = np.loadtxt(dist_path)
        if js_distance_matrix.shape != (len(sample_names), len(sample_names)):
            raise ValueError(
                f"Matrix shape {js_distance_matrix.shape} does not match number of FASTA samples {len(sample_names)}. "
                "Check that fasta_dir is the same folder/order used when computing the matrix."
            )
        js_distance_matrix = 0.5 * (js_distance_matrix + js_distance_matrix.T)
        np.fill_diagonal(js_distance_matrix, 0.0)
        js_distance_matrix[js_distance_matrix < 0] = 0.0
        jsd_matrix = js_distance_matrix ** 2
        return save_outputs(jsd_matrix, js_distance_matrix, sample_names, class_labels, metadata, args)

    if precomputed is None:
        temp_context = (
            tempfile.TemporaryDirectory(dir=args.tmp_dir)
            if args.tmp_dir is not None else tempfile.TemporaryDirectory()
        )
        with temp_context as work_dir:
            print(f"  Temporary work directory: {work_dir}", flush=True)
            print("\n[Pass 1] Streaming k-mer counts per genome...", flush=True)
            vocab_keys, raw_totals, sample_array_paths, nnz_per_sample, zero_percentage = pass1(
                files, args.k, work_dir
            )
            print(f"  Union vocabulary size: {len(vocab_keys):,}", flush=True)
            print(f"  Zero percentage before smoothing: {zero_percentage:.4f}%", flush=True)

            print("\n[Pass 2] Computing chunked JSD / JS distance matrix...", flush=True)
            jsd_matrix, js_distance_matrix, stats = compute_jsd_chunked(
                sample_array_paths,
                vocab_keys,
                raw_totals,
                args.alpha,
                args.log_base,
                args.chunk_size,
            )
            vocabulary_size = len(vocab_keys)
            del vocab_keys, raw_totals, sample_array_paths, nnz_per_sample
            gc.collect()
    else:
        vocab_keys, raw_totals, sample_array_paths, zero_percentage, work_dir = precomputed
        print(f"  Reusing k={args.k} counted arrays from: {work_dir}", flush=True)
        print("\n[Pass 2] Computing chunked JSD / JS distance matrix...", flush=True)
        jsd_matrix, js_distance_matrix, stats = compute_jsd_chunked(
            sample_array_paths,
            vocab_keys,
            raw_totals,
            args.alpha,
            args.log_base,
            args.chunk_size,
        )
        vocabulary_size = len(vocab_keys)

    return save_outputs(
        jsd_matrix,
        js_distance_matrix,
        sample_names,
        class_labels,
        metadata,
        args,
        stats=stats,
        zero_percentage=zero_percentage,
        vocabulary_size=vocabulary_size,
    )


def run_grid(args, files, sample_names, class_labels, metadata):
    if args.reuse_matrix:
        raise ValueError("--reuse_matrix is only supported for a single --k run, not --run_grid.")

    overall_rows = []

    for k in args.k_values:
        print(f"\n{'#' * 90}")
        print(f"GRID MODE: preparing k={k}")
        print(f"{'#' * 90}", flush=True)

        temp_context = (
            tempfile.TemporaryDirectory(dir=args.tmp_dir)
            if args.tmp_dir is not None else tempfile.TemporaryDirectory()
        )

        with temp_context as work_dir:
            print(f"  Temporary work directory for k={k}: {work_dir}", flush=True)
            print("\n[Pass 1] Streaming k-mer counts per genome...", flush=True)
            vocab_keys, raw_totals, sample_array_paths, nnz_per_sample, zero_percentage = pass1(
                files, k, work_dir
            )
            print(f"  Union vocabulary size for k={k}: {len(vocab_keys):,}", flush=True)
            print(f"  Zero percentage before smoothing: {zero_percentage:.4f}%", flush=True)

            args.k = k
            row = run_single(
                args,
                files,
                sample_names,
                class_labels,
                metadata,
                precomputed=(vocab_keys, raw_totals, sample_array_paths, zero_percentage, work_dir),
            )
            overall_rows.append(row)

            del vocab_keys, raw_totals, sample_array_paths, nnz_per_sample
            gc.collect()

    overall_df = pd.DataFrame(overall_rows)
    overall_path = os.path.join(args.out_dir, overall_filename())
    overall_df.to_csv(overall_path, index=False)
    print(f"\nSaved overall results CSV: {overall_path}", flush=True)

    if len(overall_df) > 0 and overall_df["overall_silhouette_score"].notna().any():
        best_idx = overall_df["overall_silhouette_score"].astype(float).idxmax()
        best = overall_df.loc[best_idx]
        print("\n" + "=" * 90)
        print("BEST SETTING")
        print("=" * 90)
        print(f"Method: {best['method']}")
        print(f"k: {best['k']}")
        print(f"alpha: {best['alpha']}")
        print(f"log base: {best['log_base']}")
        print(f"vocabulary size: {best['vocabulary_size']}")
        print(f"overall silhouette score: {best['overall_silhouette_score']:.6f}")

    return overall_df


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

    if args.run_grid:
        run_grid(args, files, sample_names, class_labels, metadata)
    else:
        row = run_single(args, files, sample_names, class_labels, metadata)
        overall_df = pd.DataFrame([row])
        overall_path = os.path.join(args.out_dir, overall_filename())
        if os.path.exists(overall_path):
            old = pd.read_csv(overall_path)
            overall_df = pd.concat([old, overall_df], ignore_index=True)
        overall_df.to_csv(overall_path, index=False)
        print(f"Saved/updated overall results CSV: {overall_path}", flush=True)


if __name__ == "__main__":
    main()