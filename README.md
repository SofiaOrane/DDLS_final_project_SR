# DDLS_final_project_SR

# Spatial Transcriptomics Analysis of Mouse Colon

## Overview
This project investigates how the presence of microbiota (Specific Pathogen-Free - SPF vs. Germ-Free - GF) alters the spatially organized transcriptomic programs in the mouse colon. Utilizing the GSE245274 dataset, we employ a modular computational pipeline (MCP) that combines unsupervised and supervised modeling techniques to understand host-microbe interactions, tissue architecture, and immune regulation. The primary goal is to identify spatially resolved molecular changes driven by the microbiota.

## Scientific Question
How does the presence of microbiota (SPF vs GF) alter the spatially organized transcriptomic programs in the mouse colon?

## Dataset
*   **Source:** GSE245274 (mouse intestine Visium spatial transcriptomics)
*   **Subset:** Colon region, SPF vs GF (approximately 6 samples, ~10,000 spots total).
*   **Content:** Count matrices (genes x spots), spot coordinates, histology images.

## Analysis Workflow (Modular Computational Pipeline - MCP)
The entire analysis pipeline is structured as a suite of modular Python scripts, each designed to perform a specific step of the analysis. These tools are located in the `mcp_scripts/` directory.

**Key Analysis Steps:**
1.  **`load_data`**: Downloads and preprocesses the GSE245274 dataset.
2.  **`eda_and_clustering`**: Performs exploratory data analysis, integrates data, and identifies spatial clusters.
3.  **`differential_expression`**: Identifies cluster-specific and condition-specific (SPF vs. GF) differentially expressed genes.
4.  **`build_graph`**: Constructs a spatial graph representation for Graph Neural Networks.
5.  **`train_gnn`**: Trains a Graph Neural Network (GNN) model for SPF vs. GF classification.
6.  **`evaluate_model`**: Evaluates the performance of the trained GNN model and generates visualizations.

## Getting Started
This project is designed to be run interactively with an AI assistant (like Gemini). The AI assistant will execute the MCP tools with the correct parameters based on your instructions.

### Prerequisites
*   Python 3.x
*   Required Python libraries (e.g., `scanpy`, `pytorch_geometric`, `numpy`, `pandas`).

### Running the Analysis
To run the analysis, simply instruct the AI assistant on which step of the pipeline you'd like to execute. For example:

*   "Run the `load_data` step."
*   "Perform EDA and clustering."
*   "Execute the differential expression analysis."
*   "Build the spatial graph."
*   "Train the GNN model."
*   "Evaluate the GNN model."

The AI assistant will automatically call the corresponding MCP tool with the necessary arguments. This was done in the GEMINI.md file. 

## Output
All generated output files, including processed data, plots, and trained models, are saved into the top-level `results_mcp/` directory. Each MCP tool creates its own subdirectory within `results_mcp/` (e.g., `results_mcp/clustering`, `results_mcp/DEA`, `results_mcp/GNN`).


## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the support provided by the DDLS Course. The analysis and report generation were significantly aided by the use of Generative AI tool Gemini CLI. 
