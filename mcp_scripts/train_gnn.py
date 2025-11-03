import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("train-gnn-mcp")

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        embeddings = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), embeddings

def train_model(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_model(model, data, mask):
    model.eval()
    out, _ = model(data)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc

@mcp.tool()
def train_gnn(
    input_graph_path: str,
    output_model_path: str,
    epochs: int = 200,
    learning_rate: float = 0.01,
    hidden_channels: int = 32,
    dropout: float = 0.5,
) -> str:
    """
    Trains a Graph Convolutional Network (GCN) model.

    Args:
        input_graph_path: Path to the input graph data (.pt file).
        output_model_path: Path to save the trained model (.pt file).
                           This should be a path like 'results_mcp/GNN/trained_model.pt'.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        hidden_channels: Number of hidden channels in the GCN.
        dropout: Dropout rate for the GCN.

    Returns:
        A message indicating the completion status and output model path.
    """
    # Load the data
    data = torch.load(input_graph_path, weights_only=False)

    # Split data based on sample
    # Using samples 0, 1, 2, 3 for training
    # Using sample 4 for validation
    # Using sample 5 for testing

    num_samples = len(np.unique(data.sample.numpy()))

    train_mask = torch.tensor([s in [0, 1, 2, 3] for s in data.sample])
    val_mask = torch.tensor([s == 4 for s in data.sample])
    test_mask = torch.tensor([s == 5 for s in data.sample])


    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=data.num_node_features, num_classes=len(data.y.unique()), hidden_channels=hidden_channels).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Train the model
    for epoch in range(epochs):
        loss = train_model(model, data, train_mask, optimizer)
        if epoch % 10 == 0:
            train_acc = test_model(model, data, train_mask)
            val_acc = test_model(model, data, val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Evaluate on the test set
    test_acc = test_model(model, data, test_mask)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)

    return f"GNN model training complete. Trained model saved to '{output_model_path}'. Test Accuracy: {test_acc:.4f}"

if __name__ == "__main__":
   mcp.run(transport="stdio")