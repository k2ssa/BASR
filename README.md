# BARS

Graph anomaly detection with budget-aware routing and selective text repair.

## 1. Installation

Requires Python 3.10+. Open PowerShell in the project root and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 2. Prepare the Data

Place the processed dataset files in the corresponding directory. For example:

```text
data/processed/Cora/
├── Cora.csv
└── Cora.mat
```

The CSV file must contain `node_id` and `text` columns. The MAT file must contain a sparse adjacency matrix named `Network` and evaluation labels named `attr_anomaly_label` and `stru_anomaly_label`. Node ordering must match across the text, graph, and labels.

For other datasets, use `Pubmed/`, `History/`, or `Arxiv/` under `data/processed/`, with filenames matching the directory name. Datasets and model weights must be obtained separately.

## 3. Configure the Services

Edit the dataset configuration, such as `configs/cora.yaml`, and set:

- `embedding.base_url`: The URL of a running OpenAI-compatible BGE embedding service.
- `embedding.model`: The model identifier accepted by the embedding service.
- `embedding.tokenizer`: The path to the local tokenizer directory matching the embedding model.
- `llm.base_url`, `llm.model`, and `llm.provider`: The settings for your LLM service.

Use the same embedding model for original documents, individual sentences, and repaired text.

## 4. Run

Set your LLM API key and start the pipeline:

```powershell
$env:LLM_API_KEY = "your-api-key"
.\.venv\Scripts\bars.exe --config configs/cora.yaml --llm-mode live --train-graph
```

The pipeline generates text embeddings, selects candidate nodes, calls the LLM for text repair, trains the graph branch, and exports detection results. API keys are read from environment variables; `.env` files are not loaded automatically.

For other datasets, use `configs/pubmed.yaml`, `configs/history.yaml`, or `configs/arxiv.yaml`. Adjust `router.budget_ratio` to change the fraction of candidate nodes; the default is `0.30`.

## 5. Outputs

Results are saved under the configured `output_root`, in a subdirectory for each dataset:

- `*_selection.csv`: Candidate nodes and repair status.
- `*_scores.csv`: Node anomaly scores. The final score is `unified_prob_or`.
- `*_metrics.csv`: AUC and AP.
