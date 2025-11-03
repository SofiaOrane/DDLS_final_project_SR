
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from mcp.server.fastmcp import FastMCP
import os
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

mcp = FastMCP("evaluate-model-mcp")

# Re-define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.embedding = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        self.embedding = x # Store the embedding from the first layer
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def test_model(model, data, mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc, pred, data.y, out

@mcp.tool()
def evaluate_model(
    input_graph_path: str,
    trained_model_path: str,
    adata_path: str, # Add adata path as input
    output_dir: str,
    hidden_channels: int = 32,
) -> str:
    """
    Evaluates a trained GNN model, generates performance metrics,
    visualizations, and saves the GNN embeddings to the AnnData object.

    Args:
        input_graph_path: Path to the input graph data (.pt file).
        trained_model_path: Path to the trained model state dictionary (.pt file).
        adata_path: Path to the integrated AnnData object (.h5ad file) to add embeddings to.
        output_dir: Directory to save the evaluation results and plots.
        hidden_channels: Number of hidden channels used in the trained GCN model.

    Returns:
        A message indicating the completion status and output location.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(input_graph_path, weights_only=False).to(device)
    adata = sc.read(adata_path)

    # Initialize model and load weights
    model = GCN(num_node_features=data.num_node_features, num_classes=len(data.y.unique()), hidden_channels=hidden_channels).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()

    # --- Evaluation on Test Set ---
    test_mask = torch.tensor([s == 5 for s in data.sample])
    accuracy, predictions, true_labels, _ = test_model(model, data, test_mask)
    f1 = f1_score(true_labels[test_mask].cpu().numpy(), predictions[test_mask].cpu().numpy(), average='weighted')

    # --- Get Embeddings for ALL data ---
    with torch.no_grad():
        _ = model(data)
        embeddings = model.embedding.cpu().numpy()

    # Add embeddings to the main AnnData object
    adata.obsm['X_gnn_embedding'] = embeddings

    # --- UMAP Visualization (similar to interpret_gnn.py) ---
    sc.pp.neighbors(adata, use_rep='X_gnn_embedding', n_neighbors=10)
    sc.tl.umap(adata)

    sc.settings.figdir = output_dir
    sc.pl.umap(adata, color='condition', save='_gnn_embedding_condition.png', title='GNN Embedding (Condition)')
    plt.close()
    sc.pl.umap(adata, color='sample', save='_gnn_embedding_sample.png', title='GNN Embedding (Sample)')
    plt.close()

    # --- UMAP Visualization of Predictions on Test Set ---
    test_indices = np.where(test_mask.cpu().numpy())[0]
    temp_adata = sc.AnnData(X=embeddings[test_indices, :])
    temp_adata.obs['true_labels'] = pd.Categorical(true_labels[test_mask].cpu().numpy())
    temp_adata.obs['predictions'] = pd.Categorical(predictions[test_mask].cpu().numpy())

    sc.pp.neighbors(temp_adata, n_neighbors=10, use_rep='X')
    sc.tl.umap(temp_adata)

    sc.pl.umap(temp_adata, color='true_labels', save='_gnn_embedding_true_labels.png', title='GNN Embedding UMAP (True Labels - Test Set)')
    plt.close()
    sc.pl.umap(temp_adata, color='predictions', save='_gnn_embedding_predictions.png', title='GNN Embedding UMAP (Predictions - Test Set)')
    plt.close()

    # Save the AnnData object with embeddings
    adata_output_path = os.path.join(output_dir, 'integrated_data_with_gnn.h5ad')
    adata.write(adata_output_path)

    results_message = (
        f"Model Evaluation Complete:\n"
        f"  Test Accuracy: {accuracy:.4f}\n"
        f"  Test F1-score (weighted): {f1:.4f}\n"
        f"Output files saved to '{output_dir}' directory.\n"
        f"AnnData with GNN embeddings saved to '{adata_output_path}'."
    )

    return results_message

if __name__ == "__main__":
   mcp.run(transport="stdio")