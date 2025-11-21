import torch
import torch.nn as nn

import os
import random
import numpy as np
from sklearn.metrics import f1_score
from collections import Counter



def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def make_weighted_ce(counts, device):
    """
    クラスの出現数から逆頻度ベースの重みを作り、CrossEntropyLoss を返す。
    少数クラスを強調する目的。
    """
    weights = 1.0 / (counts + 1e-12)   # ゼロ割り防止
    weights = weights / weights.mean()
    weights = weights.to(device=device, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device),  reduction='mean', ignore_index=-100)
    # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    return criterion



def calculate_metrics(y_pred: list, y_true: list):
    metrics = {}

    # 全体のF1スコア
    metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro')
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average='weighted')

    # 各クラスごとのF1スコア、正解数、サンプル数
    unique_classes = sorted(set(y_true) | set(y_pred))   # y_true と y_pred の両方を考慮
    class_f1_scores = f1_score(y_true, y_pred, labels=unique_classes, average=None)

    # クラスごとの正解数とサンプル数を計算
    true_counter = Counter(zip(y_true, y_pred))  # (true, pred) のペアをカウント
    total_counter = Counter(y_true)              # 各クラスのサンプル数をカウント

    for i, class_f1 in zip(unique_classes, class_f1_scores):
        metrics[f"f1_class_{i}"] = class_f1
        metrics[f"correct_num_class_{i}"] = true_counter[(i, i)]  # クラスiの正解数
        metrics[f"total_num_class_{i}"] = total_counter[i]        # クラスiのサンプル数

    return metrics