from mcp.server.fastmcp import FastMCP
import scanpy as sc
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

mcp = FastMCP("DEA-mcp")

def plot_volcano_manual(adata, output_dir):
    de_results = adata.uns['rank_genes_groups']
    group = 'SPF' # Assuming 'SPF' is the group of interest for volcano plot
    df = pd.DataFrame({
        'gene': de_results['names'][group],
        'log2fc': de_results['logfoldchanges'][group],
        'pval_adj': de_results['pvals_adj'][group]
    })
    df['-log10_pval_adj'] = -np.log10(df['pval_adj'] + 1e-300)
    df['significant'] = (df['pval_adj'] < 0.05) & (abs(df['log2fc']) > 1)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['log2fc'], df['-log10_pval_adj'], c=df['significant'].map({True: 'red', False: 'gray'}), alpha=0.5)
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10 Adjusted p-value")
    plt.title("Volcano Plot (SPF vs. GF)")
    sig_genes = df[df['significant']].sort_values(by='-log10_pval_adj', ascending=False).head(20)
    for i, row in sig_genes.iterrows():
        plt.text(row['log2fc'], row['-log10_pval_adj'], row['gene'], fontsize=9)
    plt.savefig(os.path.join(output_dir, 'volcano_plot_manual.png'))
    plt.close()

@mcp.tool()
def differential_expression(input_adata_path: str, output_dir: str) -> str:
    """
    Performs differential expression analysis on integrated spatial transcriptomics data.

    Args:
        input_adata_path: Path to the integrated AnnData object (.h5ad file).
        output_dir: Directory to save the differential expression results and plots.
                    This should be a path like 'results_mcp/DEA'.

    Returns:
        A message indicating the completion status and output location.
    """
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read(input_adata_path)

    # Perform DE analysis between clusters
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    de_clusters_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    de_clusters_df.to_csv(os.path.join(output_dir, 'de_cluster_markers.csv'))

    sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby='leiden', show=False)
    fig = plt.gcf()
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
            ax.set_title('Scaled expression')
            break
    plt.savefig(os.path.join(output_dir, 'cluster_markers_heatmap.png'), bbox_inches='tight')
    plt.close()

    dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, groupby='leiden', show=False, return_fig=True)
    dp.legend(size_title='Fraction of spots\nin group (%)', colorbar_title='Mean expression\nin group')
    dp.savefig(os.path.join(output_dir, 'cluster_markers_dotplot.png'), bbox_inches='tight')
    plt.close()

    # Perform DE analysis between conditions (SPF vs GF)
    sc.tl.rank_genes_groups(adata, 'condition', method='wilcoxon')
    de_condition_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    de_condition_df.to_csv(os.path.join(output_dir, 'de_condition_markers.csv'))

    plot_volcano_manual(adata, output_dir)

    return f"Differential expression analysis complete. Output files saved to '{output_dir}' directory."

if __name__ == "__main__":
   mcp.run(transport="stdio")