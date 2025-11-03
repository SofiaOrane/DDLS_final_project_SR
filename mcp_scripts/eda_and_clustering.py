from mcp.server.fastmcp import FastMCP
import scanpy as sc
import pandas as pd
import os
import glob
import harmonypy as hm
import matplotlib.pyplot as plt
import json

mcp = FastMCP("eda-clustering-mcp")

@mcp.tool()
def eda_and_clustering(data_dir: str, output_dir: str, target_samples_json: str) -> str:
    """
    Performs EDA, integration, and clustering on spatial transcriptomics data.

    Args:
        data_dir: Directory containing the raw 10x genomics data.
        output_dir: Directory to save the results (plots and integrated data).
                    This should be a path like 'results_mcp/clustering'.
        target_samples_json: A JSON string representing a dictionary of target samples
                             (e.g., '{"GSM7840120": "SPF", "GSM7840121": "SPF"}').

    Returns:
        A message indicating the completion status and output location.
    """
    target_samples = json.loads(target_samples_json)

    os.makedirs(output_dir, exist_ok=True)

    adatas = []

    for sample_id, condition in target_samples.items():
        print(f"Loading sample: {sample_id} ({condition})")

        sample_data_dir_base = os.path.join(data_dir, sample_id)
        subdirs = [d for d in os.listdir(sample_data_dir_base) if os.path.isdir(os.path.join(sample_data_dir_base, d))]
        if not subdirs:
            print(f"  Could not find data subdirectory for {sample_id}")
            continue
        sample_data_dir = os.path.join(sample_data_dir_base, subdirs[0])

        # Rename files if they haven't been already
        for f in glob.glob(os.path.join(sample_data_dir, "*.gz")):
            if "matrix.mtx" in f and not f.endswith("matrix.mtx.gz"):
                os.rename(f, os.path.join(sample_data_dir, "matrix.mtx.gz"))
            elif "barcodes.tsv" in f and not f.endswith("barcodes.tsv.gz"):
                os.rename(f, os.path.join(sample_data_dir, "barcodes.tsv.gz"))
            elif "features.tsv" in f and not f.endswith("features.tsv.gz"):
                os.rename(f, os.path.join(sample_data_dir, "features.tsv.gz"))

        try:
            adata = sc.read_10x_mtx(sample_data_dir, var_names='gene_symbols', cache=False)
            adata.obs['sample'] = sample_id
            adata.obs['condition'] = condition

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            adatas.append(adata)
        except Exception as e:
            print(f"  Error processing data for {sample_id}: {e}")
            continue

    if not adatas:
        return "No data loaded for processing."

    adata_merged = adatas[0].concatenate(adatas[1:], batch_key='sample_batch')

    sc.pp.highly_variable_genes(adata_merged, batch_key='sample_batch')
    adata_merged = adata_merged[:, adata_merged.var['highly_variable']]

    sc.tl.pca(adata_merged, svd_solver='arpack')

    ho = hm.run_harmony(adata_merged.obsm['X_pca'], adata_merged.obs, 'sample_batch')
    adata_merged.obsm['X_pca_harmony'] = ho.Z_corr.T

    sc.pp.neighbors(adata_merged, use_rep='X_pca_harmony', n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata_merged)

    sc.tl.leiden(adata_merged, resolution=0.5)

    sc.settings.figdir = output_dir
    sc.pl.umap(adata_merged, color='condition', save='_condition.png')
    plt.close()
    sc.pl.umap(adata_merged, color='sample_batch', save='_sample.png')
    plt.close()
    sc.pl.umap(adata_merged, color='leiden', save='_leiden_clusters.png')
    plt.close()

    adata_merged.write(os.path.join(output_dir, 'integrated_data.h5ad'))

    return f"Integration, clustering, and visualization complete. Output plots and data saved to '{output_dir}' directory."

if __name__ == "__main__":
   mcp.run(transport="stdio")