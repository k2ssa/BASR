from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import GraphLabels


def evaluate(scores: dict[str, np.ndarray], labels: GraphLabels) -> pd.DataFrame:
    truths = {
        "attr_anomaly": labels.attribute,
        "stru_anomaly": labels.structure,
        "unified_anomaly": labels.unified,
    }
    rows = []
    for label_name, truth in truths.items():
        # Real-world datasets may provide only one unified anomaly label.  AUC
        # is undefined for an all-zero/all-one task, so omit such rows instead
        # of fabricating duplicate attribute/structure labels.
        if np.unique(truth).size < 2:
            continue
        for signal, score in scores.items():
            rows.append(
                {
                    "label": label_name,
                    "signal": signal,
                    "n_pos": int(truth.sum()),
                    "AUC": round(float(roc_auc_score(truth, score)), 4),
                    "AP": round(float(average_precision_score(truth, score)), 4),
                }
            )
    return pd.DataFrame(rows)
