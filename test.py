import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

# from model.test_model import Model
from model.model import Model
from utils.utility import set_seed
from utils.dataset import Dataset
from utils.collate_fn import CollateFn
from utils.utility import calculate_metrics

tqdm.write(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--speaker_state_dim", type=int)
    parser.add_argument("--pause_dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--local_window_num", type=int)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--trained_filename", type=str)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        speaker_state_dim=args.speaker_state_dim,
        pause_dim=args.pause_dim,
        heads=args.heads,
        local_window_num=args.local_window_num,
        dropout_rate=args.dropout_rate,
        trained_filename=args.trained_filename
    )

    # モデル全体をGPUに移動
    model = model.to(device)

    # データセットとデータローダーの準備
    test_dataset = Dataset(dataset=args.dataset, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollateFn())

    label_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for conv in test_dataset:
        for utt in conv:
            lbl = utt["label"]
            if lbl != -100:
                label_num[lbl] += 1
    print("Test set label distribution:", label_num)


    # ===== TEST =====
    model.eval()
    pred = []
    true = []
    
    # softmax(logits)を保存するためのリスト
    # softmax_for_label_1 = []
    # softmax_for_label_4 = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            time_mask = batch["time_mask"].to(device)   # (B, T_max)
            utt_mask = batch["utt_mask"].to(device)
            pauses = batch["pauses"].to(device)
            speakers = batch["speakers"].to(device)
            labels = batch["labels"].to(device)

            _, T_max, _ = input_ids.size()
            test_cache = None

            for t in range(T_max):
                input_ids_t = input_ids[:, t, :].contiguous()   # (B, U_max)
                utt_mask_t = utt_mask[:, t, :].contiguous()     # (B, U_max)
                pause_t = pauses[:, t].contiguous()             # (B)
                label_t = labels[:, t].contiguous()             # (B)

                with autocast():
                    logits, global_x = model(t, input_ids_t, time_mask, utt_mask_t, pause_t, test_cache, speakers)

                # cache更新
                if (test_cache is None):
                    test_cache = global_x.unsqueeze(1)   # (B, 1, hidden)
                else:
                    test_cache = torch.cat([test_cache, global_x.unsqueeze(1)], dim=1)   # (B, t+1, hidden)

                # evaluate only valid labels (ignore -100)
                valid_idx = (label_t != -100)
                if (valid_idx.any()):
                    valid_logits = logits[valid_idx]
                    valid_labels = label_t[valid_idx]

                    # # softmaxを計算
                    # softmax_probs = torch.softmax(valid_logits, dim=1)
                    # # ラベルが1のときのsoftmax値を取得
                    # indices_label_1 = (valid_labels == 1).nonzero(as_tuple=True)[0]
                    # if (indices_label_1.numel() > 0):
                    #     softmax_for_label_1.append(softmax_probs[indices_label_1].cpu())
                    # # ラベルが4のときのsoftmax値を取得
                    # indices_label_4 = (valid_labels == 4).nonzero(as_tuple=True)[0]
                    # if (indices_label_4.numel() > 0):
                    #     softmax_for_label_4.append(softmax_probs[indices_label_4].cpu())

                    pred_labels = torch.argmax(valid_logits, dim=1).cpu().tolist()
                    true_labels = valid_labels.cpu().tolist()
                    pred.extend(pred_labels)
                    true.extend(true_labels)

    metrics = calculate_metrics(pred, true)
    macro_f1 = metrics['f1_macro']
    weighted_f1 = metrics['f1_weighted']
    tqdm.write(f"Macro F1 Score: {macro_f1:.4f}")
    tqdm.write(f"Weighted F1 Score: {weighted_f1:.4f}") 
    for i in range(args.num_classes):
        tqdm.write(f"{metrics[f'correct_num_class_{i}']} / {metrics[f'total_num_class_{i}']}  =>  f1: {metrics[f'f1_class_{i}']:.4f}")

    # 学習可能閾値を確認1
    # if (args.pause_dim > 0):
    #     tqdm.write(f"Learned time threshold: {model.time_threshold.item():.4f}")


if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        tqdm.write(f"{arg}: {getattr(_args, arg)}")
    set_seed(_args.seed)
    train(_args)