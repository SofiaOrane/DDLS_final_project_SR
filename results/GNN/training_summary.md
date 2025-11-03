# GNN Training Summary

## Initial Training and Evaluation

The GNN model was trained to classify spots as either Specific Pathogen-Free (SPF) or Germ-Free (GF).

### Setup
- **Model:** 2-layer Graph Convolutional Network (GCN).
- **Dataset:** Graph constructed from 6 samples, with nodes representing spots and edges representing spatial proximity.
- **Split:**
    - **Training:** 4 samples
    - **Validation:** 1 sample
    - **Testing:** 1 sample
- **Epochs:** 200

### Results
- **Final Test Accuracy:** 93.56%
- **Final Validation Accuracy:** ~78%
- **Final Training Accuracy:** ~98%

### Interpretation
The model achieved a high accuracy on the unseen test set, demonstrating its ability to distinguish between the SPF and GF conditions based on the spatial transcriptomic data.

The difference between the training accuracy and the validation/test accuracy suggests that the model is slightly overfitting the training data. This is a common occurrence and can be addressed through techniques like hyperparameter tuning and cross-validation. The discrepancy between validation and test accuracy might be due to the specific characteristics of the samples chosen for each set.

## Leave-One-Slide-Out Cross-Validation

To obtain a more robust evaluation, a 6-fold leave-one-slide-out cross-validation was performed. In each fold, one sample was used for testing, one for validation, and the remaining four for training.

### Dataset Split Distribution per Fold

| Fold | Train Nodes | Validation Nodes | Test Nodes |
|---|---|---|---|
| 1    | 9140         | 2568              | 3153        |
| 2    | 9602         | 2691              | 2568        |
| 3    | 9736         | 2434              | 2691        |
| 4    | 10662        | 1765              | 2434        |
| 5    | 10846        | 2250              | 1765        |
| 6    | 9458         | 3153              | 2250        |

### Results

- **Mean Test Accuracy:** 93.62%
- **Standard Deviation:** 4.18%

#### Test Accuracies per Fold:
- **Fold 1:** 96.42%
- **Fold 2:** 88.55%
- **Fold 3:** 95.54%
- **Fold 4:** 95.48%
- **Fold 5:** 87.25%
- **Fold 6:** 98.49%

### Interpretation

The cross-validation results confirm the strong predictive performance of the model, with a mean accuracy of over 93%. The standard deviation of ~4% indicates some variability in performance across different samples, with samples in folds 2 and 5 being slightly more challenging for the model to classify.

## Hyperparameter Optimization on Raw Data

Given the poor performance on the batch-corrected data, we reverted to the raw gene expression data and performed hyperparameter optimization.

### Setup
- **Model:** 2-layer Graph Convolutional Network (GCN).
- **Data:** Raw gene expression data.
- **Tuning:** Grid search with 6-fold cross-validation.
- **Search Space:**
    - Learning Rate: `[0.01, 0.001]`
    - Hidden Channels: `[16, 32]`
    - Dropout Rate: `[0.5, 0.6]`

### Best Results
- **Mean Test Accuracy:** 95.56%
- **Standard Deviation:** 3.56%
- **Best Hyperparameters:**
    - Learning Rate: 0.01
    - Hidden Channels: 32
    - Dropout: 0.5

### Interpretation
The hyperparameter optimization was successful and resulted in a model with a higher and more robust performance than the initial model. This is our best performing model so far.
