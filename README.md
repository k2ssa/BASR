# BARS

预算感知的文本属性图异常检测：句子路由 → 选择性 LLM 反事实修复 → 属性证据，与 ANEMONE 结构感知分支进行经验秩归一化和 Probability-OR 融合。

本仓库只包含主方法。Python 包名保留为 `llm_anemone`。

## 代码职责

| 文件 | 用途 |
|---|---|
| `__main__.py`、`cli.py`、`config.py` | 命令入口、参数覆盖、配置与环境变量 |
| `pipeline.py` | 端到端组装、缓存、训练调用与结果输出 |
| `data.py` | CSV 文本、MAT 邻接矩阵与评估标签读取 |
| `embeddings.py` | 原文、句子和修复文本的 BGE API 编码与缓存 |
| `routing.py` | 主配置 `adaptive_internal` 预算路由 |
| `llm/`、`prompts/` | 提示词、LLM 在线推理、断点续跑和响应解析 |
| `anemone/` | 模型、随机游走子图采样、训练和异常评分 |
| `scoring.py`、`utils.py` | 属性证据、经验秩和双分支融合、随机种子和设备选择 |
| `evaluation.py` | AUC/AP；标签只在最终评估和导出时使用 |
| `configs/` | Cora、Pubmed、History、Arxiv 四个主方法配置 |

主配置：30% 节点预算、`adaptive_internal`、DeepSeek-V4-Flash、属性证据 `0.9 × semantic_change + 0.1 × neighbor_gain`，最终读取 `unified_prob_or` 分数。

Prompt 按现有运行配置保留：Cora/Pubmed/Arxiv 使用 `teacher_v9.yaml`，History 使用面向历史图书文本的 `teacher_history_v9.yaml`，不是完全相同的提示词。

## 安装

在本仓库根目录执行（Python 3.10+，以下为 PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m llm_anemone --help
```

`requirements.txt` 固定本机已安装的直接依赖版本；本机 PyTorch 是 `2.12.0.dev20260408+cu128`，发布版只声明 `torch>=2.6`，不强制安装该 nightly CUDA 构建。这不是完整环境锁文件，未验证所有 PyTorch/Python 组合。

## 数据准备

数据、权重、LLM 响应和实验结果不包含在代码仓库中，需要单独准备。目录约定：

```text
data/processed/
  Cora/Cora.csv, Cora.mat
  Pubmed/Pubmed.csv, Pubmed.mat
  History/History.csv, History.mat
  Arxiv/Arxiv.csv, Arxiv.mat
```

每个 CSV 必须包含 `node_id` 和 `text` 列。MAT 需包含稀疏矩阵 `Network`，以及 `attr_anomaly_label`、`stru_anomaly_label`；可额外提供 `unified_anomaly_label`。CSV 行顺序必须与图矩阵和标签的节点顺序一致。

加载器优先读取同目录的 `<Dataset>_repaired.mat`（如果存在）。复现使用恢复标签的 Arxiv 实验时，也需准备对应修复文件。本仓库不包含从原始数据构造异常实例或恢复标签的旧实验脚本，不能直接用任意同名原始数据替代论文中的异常实例。

## BGE 编码服务

本代码是 OpenAI 兼容 embeddings API 的客户端；需先自行准备编码服务和对应的本地 tokenizer。

配置中的 `embedding.base_url` 指向服务，`embedding.model` 是服务端接受的模型标识，`embedding.tokenizer` 是本机 tokenizer 文件夹路径。相对路径从 `project_root` 解析；可以改为绝对路径。原文、句子、修复文本必须使用相同模型与编码规则。

原配置保留 `/models/bge-m3-en-v1.5` 和 `model/bge-m3-en-v1.5` 作为历史服务标识与目录约定；目录名不等于模型身份。本机该目录的配置显示 768 维 BERT、来源路径含 `BAAI_bge-base-en`。论文实验记录使用 BGE-base；复现前需核对服务实际加载的权重和 tokenizer，不能仅凭目录名选 BGE-M3。

可在 PowerShell 覆盖服务地址与模型：

```powershell
$env:EMBED_BASE_URL = "http://localhost:8000/v1"
$env:EMBED_MODEL = "/models/bge-m3-en-v1.5"
$env:EMBED_API_KEY = "EMPTY"
```

## 首次运行

先修改 `configs/cora.yaml` 的 embedding 服务及 tokenizer 路径，准备好上述数据，然后：

```powershell
$env:LLM_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m llm_anemone --config configs/cora.yaml --llm-mode live --train-anemone
```

这条命令会调用编码服务、对预算内候选调用在线 LLM，并训练 ANEMONE。也可使用 `DEEPSEEK_API_KEY`。`.env.example` 仅供参考，程序不会自动加载 `.env`。

其他数据集分别使用 `configs/pubmed.yaml`、`configs/history.yaml`、`configs/arxiv.yaml`。配置保留原实验使用的模型标识；接入其他可用模型时，可通过 `LLM_BASE_URL`、`LLM_MODEL` 覆盖，并确认 `llm.provider` 与服务兼容。更换模型后应更换 `llm.output_tag`，避免混用缓存；模型变更不代表复现了原论文结果。

`--train-anemone` 会强制训练；首次运行必须提供。后续复用结构分数时，把配置的 `anemone.score_file` 设为实际生成的分数 CSV，再去掉该开关。修改训练设置后需重新训练，不能直接复用旧分数。

## 缓存与输出

原文 embedding 缓存在 `data/processed/<Dataset>/<Dataset>_exp_x0_emb.pt`；其他输出位于 `outputs/llm_anemone/<Dataset>/`。更换编码器时，首次加上 `--force-node-embeddings --force-sentence-embeddings --force-repair-embeddings` 重建缓存。

结果包括节点选择表 `*_selection.csv`、异常分数表 `*_scores.csv` 和指标表 `*_metrics.csv`。ANEMONE 同时输出权重和结构分数。

已有完整且匹配的原文/句子 embedding、routed LLM JSONL、修复 embedding 以及 ANEMONE 分数时，可设置 `anemone.score_file`，再使用 `--llm-mode routed_cache` 离线复算。缓存不完整时仍可能调用编码服务或报缺文件错误。`cache` 模式读取另一套历史全图缓存，首次运行请使用 `live`。

## 发布范围与来源

此精简版从工作目录中按命令入口的静态导入依赖提取。未包含旧 notebook、旧 GCN/MLP/DOMINANT 流程、对比方法源码、消融/扫参/绘图脚本、数据、预训练模型、API 密钥、实验输出或 Git 历史。

检测算法保持原样，仅将默认项目路径改为当前目录、数据路径改为 `data/processed`，并将配置中指向本机已有结果的 `anemone.score_file` 改为 `null`。

ANEMONE 是已有方法，本项目在其架构基础上实现结构感知分支；BGE 与 LLM 为外部模型/服务。
