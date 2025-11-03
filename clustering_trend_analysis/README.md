# Clustering and Identification of Trends in Summarised PDF Data

This Proof of Concept (POC) ingests JSON summaries of papers/articles, generates sentence embeddings, clusters similar items, reduces dimensions to 2D, and visualizes clusters and yearly trends. Outputs include a clustered CSV and interactive Plotly HTML files.

## Project Structure

```
clustering_trend_analysis/
├── data/
│   └── sample_data.json
├── outputs/                 # generated artifacts (CSV + HTML plots)
├── src/
│   ├── load_data.py         # load_json_to_df
│   ├── embeddings.py        # generate_embeddings
│   ├── clustering.py        # run_clustering_and_pca
│   ├── visualization.py     # create_interactive_visualizations
│   └── main.py              # CLI orchestration
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```

Note: `sentence-transformers` will download the model on first run. Ensure internet access for that step.

## Input Format

`data/sample_data.json` is a list of records with the following fields:

- `title` (string)
- `year` (int)
- `problem` (string)
- `findings` (string)
- `limitations` (string)

Example:

```json
[
  {
    "title": "Deep Learning for Medical Imaging",
    "year": 2019,
    "problem": "Automating disease detection in radiology images",
    "findings": "CNNs achieve high accuracy on several benchmark datasets",
    "limitations": "Limited interpretability and generalization across institutions"
  }
]
```

## Running

From the project root:

```bash
python -m clustering_trend_analysis.src.main \
  --data clustering_trend_analysis/data/sample_data.json \
  --output-dir clustering_trend_analysis/outputs \
  --clusters 3 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 32 \
  --verbosity 1
```

On completion, you will find:
- `outputs/clustered_results.csv`
- `outputs/clusters_scatter.html`
- `outputs/yearly_trend.html`

Open the HTML files in a browser for interactive exploration.

## Notes
- Text for embeddings is the concatenation of `problem`, `findings`, and `limitations`.
- KMeans is used for clustering; PCA (2D) is used only for visualization.
- You can adjust the number of clusters with `--clusters`.
- If running on CPU, expect the first embedding pass to take longer due to model download.

