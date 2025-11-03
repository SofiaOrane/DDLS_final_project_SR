# Project Decisions Log

## 2025-10-21

### 1. Batch Effect Observed in GNN Embeddings

- **Observation:** After the initial successful GNN training (93.6% accuracy), the UMAP visualization of the node embeddings showed a clear clustering by sample, indicating a strong batch effect.
- **Concern:** The model might be learning sample-specific features instead of the biological condition, making the high accuracy misleading.

### 2. Diagnosis: GNN Not Using Batch-Corrected Data

- **Cause:** The `build_graph.py` script was using the raw gene expression data (`adata.X`) for the GNN's node features, not the Harmony-corrected data (`adata.obsm['X_pca_harmony']`).
- **Decision:** Modify `build_graph.py` to use the Harmony-corrected data for both graph construction and node features.

### 3. Re-evaluation with Batch-Corrected Data

- **Action:** Re-ran the cross-validation (`train_gnn_cv.py`) with the corrected data.
- **Result:** The mean test accuracy dropped drastically to ~37%.
- **Interpretation:** The GNN model and its current hyperparameters are not suitable for the new, lower-dimensional (PCA-based) and batch-corrected input data.

### 4. Hyperparameter Optimization on Corrected Data

- **Action:** Performed hyperparameter optimization on the model using the corrected data.
- **Result:** The best mean accuracy achieved was ~40%.
- **Interpretation:** Even with HPO, the model performance on the corrected data is poor.

### 5. Reverting to Raw Gene Expression Data

- **Decision:** Due to the poor performance on the batch-corrected data, the user has decided to revert to using the raw gene expression data. The new strategy is to optimize the model that works directly with the raw data, while acknowledging the potential influence of the batch effect.
- **New Plan:**
    1. Revert `build_graph.py` to use the raw gene expression data.
    2. Re-run `build_graph.py`.
    3. Perform hyperparameter optimization on the model using the raw gene expression data.

### 6. Hyperparameter Optimization on Raw Data

- **Action:** Performed hyperparameter optimization on the model using the raw gene expression data, with full cross-validation.
- **Result:** The best model achieved a mean cross-validation accuracy of **95.56%**.
- **Best Hyperparameters:**
    - Learning Rate: 0.01
    - Hidden Channels: 32
    - Dropout: 0.5
- **Interpretation:** The HPO was successful in finding a better set of parameters, leading to a significant improvement in the model's performance on the raw data.
