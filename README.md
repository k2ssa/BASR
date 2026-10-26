# BARS

基于预算路由与选择性文本修复的图异常检测。

## 1. 安装

需要 Python 3.10+。在项目根目录打开 PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 2. 准备数据

将处理好的数据放到对应目录，例如：

```text
data/processed/Cora/
├── Cora.csv
└── Cora.mat
```

CSV 包含 `node_id`、`text` 两列；MAT 包含稀疏邻接矩阵 `Network` 和评估标签 `attr_anomaly_label`、`stru_anomaly_label`。文本、图和标签的节点顺序必须一致。

其他数据集按同样方式放到 `Pubmed/`、`History/`、`Arxiv/`，文件名与目录名一致。数据和模型权重需单独准备。

## 3. 修改配置

以 `configs/cora.yaml` 为例，设置：

- `embedding.base_url`：已启动的 OpenAI 兼容 BGE 编码服务地址。
- `embedding.model`：编码服务实际加载的模型标识。
- `embedding.tokenizer`：与编码模型匹配的本地 tokenizer 文件夹路径。
- `llm.base_url`、`llm.model`、`llm.provider`：使用的 LLM 服务配置。

原文、句子和修复文本应使用相同的编码模型。

## 4. 运行

设置 LLM 密钥并启动：

```powershell
$env:LLM_API_KEY = "你的API密钥"
.\.venv\Scripts\bars.exe --config configs/cora.yaml --llm-mode live --train-graph
```

程序会生成文本表示、选择候选节点、调用 LLM 修复、训练图分支并输出检测结果。密钥通过环境变量传入，程序不会自动读取 `.env`。

运行其他数据集时，将配置路径换成 `configs/pubmed.yaml`、`configs/history.yaml` 或 `configs/arxiv.yaml`。调整 `router.budget_ratio` 可修改候选节点比例，默认是 `0.30`。

## 5. 查看结果

结果保存在配置的 `output_root` 下，按数据集分目录：

- `*_selection.csv`：候选节点及修复状态。
- `*_scores.csv`：节点异常分数，最终分数为 `unified_prob_or`。
- `*_metrics.csv`：AUC 和 AP。
