## Data Integration and Clustering Plan

1.  **Load and Prepare Data:** Load each of the 6 samples into a separate `AnnData` object, which is the standard data structure used by `scanpy`.

2.  **Normalize Each Sample:** For each sample, I will:
    *   Normalize the total counts per spot to a common scale.
    *   Apply a logarithmic transformation.

3.  **Identify Highly Variable Genes (HVGs):** I will identify the most variable genes within each sample. These genes are the most likely to be biologically interesting.

4.  **Integrate Data with Harmony:** I will use the Harmony algorithm to integrate the data from the 6 samples. Harmony is a robust method that corrects for batch effects, allowing for a more accurate comparison between the SPF and GF conditions.

5.  **Clustering and Visualization:** After integration, I will:
    *   Perform unsupervised clustering on the corrected data to identify spatial domains.
    *   Visualize the results using UMAP (Uniform Manifold Approximation and Projection) to see how the spots from different samples and conditions group together.
