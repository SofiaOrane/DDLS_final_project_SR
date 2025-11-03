import scanpy as sc
import torch
from torch_geometric.data import Data
from mcp.server.fastmcp import FastMCP
import os
from sklearn.preprocessing import LabelEncoder

mcp = FastMCP("build-graph-mcp")

@mcp.tool()
def build_graph(
    adata_path: str,
    output_dir: str,
    n_top_genes: int = 2000,
    n_neighbors: int = 15,
    n_pcs: int = 50,
) -> str:
    """
    Builds a spatial graph from the AnnData object.

    This function takes a path to an AnnData object, performs highly variable
    gene selection, computes a neighborhood graph, and saves the resulting
    graph in a PyTorch Geometric Data object.

    Args:
        adata_path: Path to the input AnnData object (.h5ad file).
        output_dir: Directory to save the output graph data.
                    This should be a path like 'results_mcp/GNN'.
        n_top_genes: Number of highly variable genes to use.
        n_neighbors: Number of neighbors to use for the graph construction.
        n_pcs: Number of principal components to use for the neighbor search.

    Returns:
        A string indicating the path to the saved graph data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the AnnData object
    adata = sc.read(adata_path)

    # 1. Select highly variable genes
    # This is a crucial step to focus on the most informative genes and
    # reduce computational complexity.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable]

    # 2. Build the graph using scanpy's neighbors function
    # We use the gene expression data ('X') to compute the graph.
    # `n_pcs` are used to reduce the dimensionality of the data before
    # computing the nearest neighbors.
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X')

    # 3. Convert the graph to a PyTorch Geometric Data object
    #    - Get the adjacency matrix from adata.obsp['connectivities']
    #    - Get the node features from adata.X
    #    - Get the node labels (condition) from adata.obs['condition']

    # Extract edge index from the adjacency matrix
    adj = adata.obsp['connectivities'].tocoo()
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)

    # Extract node features
    x = torch.tensor(adata.X.toarray(), dtype=torch.float)

    # Encode labels
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(adata.obs['condition']), dtype=torch.long)

    # Encode samples
    sample_le = LabelEncoder()
    sample = torch.tensor(sample_le.fit_transform(adata.obs['sample']), dtype=torch.long)

    # Create the Data object
    data = Data(x=x, edge_index=edge_index, y=y, sample=sample)

    # 4. Save the Data object to a file
    output_path = os.path.join(output_dir, "graph_data.pt")
    torch.save(data, output_path)

    return f"Graph data saved to {output_path}"

if __name__ == "__main__":
    mcp.run(transport="stdio")