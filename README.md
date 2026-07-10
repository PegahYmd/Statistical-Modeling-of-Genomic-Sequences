# Statistical Modeling of Genomic Sequences

**Author:** Pegah Yarahmadi  
**Institution:** Politecnico di Torino - Master's in Artificial Intelligence  
**Supervisors:** Prof. Renato Ferrero, Dr. Chiara Panico  

---

## Overview
This repository contains the codebase and experimental data for my Master's thesis. The research focuses on the statistical modeling of genomic sequences using information-theoretic measures. By analyzing k-mer probability distributions, this project implements and evaluates various sequence alignment-free algorithms to determine genomic similarities and distances.

## Repository Structure
The codebase is modularized based on the specific mathematical approaches and datasets used during the evaluation phases:

*   **`/ASH1L DNA samples`**: DNA samples resulted in combination of the human ASH1L gene sequences and variation of functional and pathological types.
*   **`/normal KL`**: Scripts computing the standard Kullback-Leibler (KL) divergence between k-mer distributions.
*   **`/weighted KL`**: Advanced KL divergence implementation utilizing a 4 different weight functions. The novel solution developed for the topic.
*   **`/JSD`**: Modules for calculating the Jensen-Shannon Distance.
*   **`/Cosine similarity`**: Cosine similarity-based k-mer comparison implementations.
*   **`/Kwip`**: Scripts executing kWIP-style entropy-weighted inner product computations.
*   **`/phase2-Brucella dataset experiment`**: The secondary evaluation phase, containing the comparative analysis and distance matrix calculations for the Brucella genomic dataset (which includes samples with a "clear" status).

## Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, Scikit-learn (for evaluation metrics like silhouette scores), NumPy.
*   **Documentation:** LaTeX is utilized for generating thesis result tables from the computational outputs.