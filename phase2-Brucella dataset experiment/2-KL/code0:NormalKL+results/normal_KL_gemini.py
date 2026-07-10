import numpy as np
import glob
import os
import argparse
import pickle
import gc
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples

# =========================================================
# COMMAND LINE ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(description="Disk-Backed Sparse KL Divergence")
parser.add_argument("-k", type=int, required=True, help="k-mer size (e.g., 15, 21, 31)")
parser.add_argument("--alpha", type=float, default=0.1, help="Smoothing parameter (default: 0.1)")
parser.add_argument("--pattern", type=str, default="../1-samples/*.fa", help="Path pattern to FASTA files")
args = parser.parse_args()

K_VAL = args.k
ALPHA_VAL = args.alpha
FASTA_PATTERN = args.pattern
METHOD_NAME = "baseline_KL_disk_sparse"

# Create a temporary directory to hold our dictionary files
CACHE_DIR = f"kmer_cache_k{K_VAL}"
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================================================
# FASTA READER & K-MER COUNTER
# =========================================================

def read_fasta(path):
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line.upper())
    return "".join(seq)

def count_kmers(sequence, k):
    counts = Counter()
    valid_chars = set("ACGT")
    # Using a simple loop is fine since we process one at a time now
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if set(kmer).issubset(valid_chars):
            counts[kmer] += 1
    return counts

# =========================================================
# SPARSE KL DIVERGENCE CALCULATION
# =========================================================

def sparse_kl_divergence(counts_P, counts_Q, S_P, S_Q, V, alpha):
    denom_P = S_P + alpha * V
    denom_Q = S_Q + alpha * V
    p_default = alpha / denom_P
    q_default = alpha / denom_Q
    kl_sum = 0.0
    
    keys_P_or_Q = set(counts_P.keys()).union(counts_Q.keys())

    for kmer in keys_P_or_Q:
        c_p = counts_P.get(kmer, 0)
        c_q = counts_Q.get(kmer, 0)
        p = (c_p + alpha) / denom_P
        q = (c_q + alpha) / denom_Q
        kl_sum += p * np.log(p / q)

    num_unseen = V - len(keys_P_or_Q)
    if num_unseen > 0:
        unseen_contribution = num_unseen * (p_default * np.log(p_default / q_default))
        kl_sum += unseen_contribution

    return kl_sum

# =========================================================
# INITIALIZE FILES & EXTRACT LABELS 
# =========================================================

files = sorted(glob.glob(FASTA_PATTERN))
if len(files) == 0:
    raise FileNotFoundError(f"No FASTA files found using pattern: {FASTA_PATTERN}")

sample_names = [os.path.basename(f) for f in files]
N = len(files)
print(f"Loaded {N} FASTA files.")

label_names = []
for f in files:
    with open(f, "r") as file:
        header = file.readline().strip()
        parts = header.split()
        if len(parts) >= 4 and parts[2].lower() == "sp.":
            species = f"{parts[1]} {parts[2]} {parts[3]}"
        elif len(parts) >= 3:
            species = f"{parts[1]} {parts[2]}"
        else:
            species = "Unknown Brucella"
        label_names.append(species)

unique_classes = sorted(list(set(label_names)))
class_to_id = {name: idx for idx, name in enumerate(unique_classes)}
labels = np.array([class_to_id[name] for name in label_names])

# =========================================================
# DISK-CACHING K-MER PROCESSING (OOM-PROOF)
# =========================================================

print("\n" + "=" * 60)
print(f"Phase 1: Counting and Caching k-mers to disk (k={K_VAL})")
print("=" * 60)

global_vocab_hashes = set()
total_sums = []
cache_paths = []

for idx, f in enumerate(files):
    print(f"  Processing {idx+1}/{N}: {sample_names[idx]}...")
    
    # Read sequence
    seq = read_fasta(f)
    
    # Count k-mers
    c = count_kmers(seq, K_VAL)
    
    # Calculate sum and add integer hashes to global vocabulary
    total_sums.append(sum(c.values()))
    global_vocab_hashes.update(hash(kmer) for kmer in c.keys())
    
    # Save the dictionary to disk
    cache_path = os.path.join(CACHE_DIR, f"sample_{idx}.pkl")
    with open(cache_path, 'wb') as pkl_file:
        pickle.dump(c, pkl_file)
    cache_paths.append(cache_path)
    
    # MANUALLY CLEAR RAM
    del seq
    del c
    gc.collect()

V = len(global_vocab_hashes)
del global_vocab_hashes
gc.collect()

print(f"\nPhase 1 Complete. Global Vocabulary size (V) = {V}")

# =========================================================
# COMPUTE PAIRWISE DIVERGENCE (LOADING ON DEMAND)
# =========================================================

print("\n" + "=" * 60)
print(f"Phase 2: Computing pairwise divergence matrix")
print("=" * 60)

D_kl_directed = np.zeros((N, N), dtype=np.float64)

for i in range(N):
    # Load Genome P from disk once for the row
    with open(cache_paths[i], 'rb') as f_P:
        counts_P = pickle.load(f_P)
        
    for j in range(N):
        if i != j:
            # Load Genome Q from disk
            with open(cache_paths[j], 'rb') as f_Q:
                counts_Q = pickle.load(f_Q)
                
            D_kl_directed[i, j] = sparse_kl_divergence(
                counts_P, counts_Q, 
                total_sums[i], total_sums[j], 
                V, ALPHA_VAL
            )
            del counts_Q
            
    del counts_P
    gc.collect()

D_kl_symmetric = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    for j in range(N):
        D_kl_symmetric[i, j] = 0.5 * (D_kl_directed[i, j] + D_kl_directed[j, i])

np.fill_diagonal(D_kl_symmetric, 0)
D_kl_symmetric[D_kl_symmetric < 0] = 0

# =========================================================
# SILHOUETTE SCORE & CSV OUTPUT
# =========================================================

try:
    if len(unique_classes) > 1:
        overall_sil = silhouette_score(D_kl_symmetric, labels, metric="precomputed")
        print(f"\nOverall Silhouette Score: {overall_sil:.4f}")
    else:
        overall_sil = float('nan')
        print("\nSkipped Silhouette Score: Only one class detected.")
except ValueError as e:
    overall_sil = float('nan')
    print(f"\nCould not compute Silhouette Score: {e}")

output_csv = "disk_sparse_KL_summary.csv"
file_exists = os.path.isfile(output_csv)

with open(output_csv, "a") as f:
    if not file_exists:
        f.write("method,k,alpha,vocabulary_size,overall_silhouette_score\n")
    f.write(f"{METHOD_NAME},{K_VAL},{ALPHA_VAL},{V},{overall_sil:.8f}\n")

print(f"Saved results to {output_csv}")