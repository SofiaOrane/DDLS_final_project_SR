# Project: Spatial Transcriptomics Analysis of Mouse Colon

## Project Context
This project aims to investigate how the presence of microbiota (Specific Pathogen-Free - SPF vs. Germ-Free - GF) alters the spatially organized transcriptomic programs in the mouse colon. We will utilize the GSE245274 dataset, employing both unsupervised and supervised modeling techniques to understand host–microbe interactions, tissue architecture, and immune regulation.

## Scientific Question
How does the presence of microbiota (SPF vs GF) alter the spatially organized transcriptomic programs in the mouse colon?

## Dataset
**Source:** GSE245274 (mouse intestine Visium spatial transcriptomics)
**Subset:** Colon region, SPF vs GF (approximately 6 samples, ~10,000 spots total).
**Content:** Count matrices (genes × spots), spot coordinates, histology images.

## Analysis Workflow using MCP Tools
The entire analysis pipeline has been modularized into a suite of MCP (Modular Computational Pipeline) tools. Each step is implemented as a separate Python script within the `mcp_scripts/` directory and is designed to be callable from the command line.

**Key Analysis Steps:**

1.  **`load_data`**: Downloads and preprocesses the GSE245274 dataset, including initial filtering and normalization.
    *   **Tool:** `mcp_scripts/load_data.py`

2.  **`eda_and_clustering`**: Performs exploratory data analysis, integrates data from multiple samples, and identifies spatial clusters within the colon tissue.
    *   **Tool:** `mcp_scripts/eda_and_clustering.py`

3.  **`differential_expression`**: Identifies cluster-specific and condition-specific (SPF vs. GF) differentially expressed genes.
    *   **Tool:** `mcp_scripts/differential_expression.py`

4.  **`build_graph`**: Constructs a spatial graph representation where spots are nodes and spatial proximity defines edges, preparing data for Graph Neural Networks.
    *   **Tool:** `mcp_scripts/build_graph.py`

5.  **`train_gnn`**: Trains a Graph Neural Network (GNN) model (e.g., GCN/GraphSAGE) for SPF vs. GF classification of Visium spots.
    *   **Tool:** `mcp_scripts/train_gnn.py`

6.  **`evaluate_model`**: Evaluates the performance of the trained GNN model using metrics such as accuracy, F1-score, and ROC-AUC, and generates relevant visualizations.
    *   **Tool:** `mcp_scripts/evaluate_model.py`

## Output Directory
All generated output files, including processed data, plots, and trained models, **must be saved into a new top-level directory named `results_mcp/`**. Each MCP tool will expect an `output_dir` argument where you should specify a path within this `results_mcp/` structure (e.g., `results_mcp/clustering`, `results_mcp/DEA`, `results_mcp/GNN`, `results_mcp/GNN_evaluation`).

## How to Use the MCP Tools
These MCP tools are designed to be run by me, Gemini, your AI assistant. When you start a new session and want to run the analysis, simply instruct me on which step of the pipeline you\'d like to execute. I will automatically run the corresponding MCP tool with the correct parameters. You do not need to manually run any `mcp run...` commands.