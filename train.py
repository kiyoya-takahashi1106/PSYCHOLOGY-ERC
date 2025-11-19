import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import os
import argparse
from tqdm import tqdm

from model.model import Model
from utils.utility import set_seed
from utils.dataset import Dataset
from utils.collate_fn import CollateFn
from utils.utility import make_weighted_ce
from utils.utility import calculate_metrics

tqdm.write(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--roberta_lr", type=str)
    parser.add_argument("--else_lr", type=str)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--speaker_state_dim", type=int)
    parser.add_argument("--time_dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--local_window_num", type=int)
    parser.add_argument("--dropout_rate", type=float)
    args = parser.parse_args()
    return args


def train(args):
    exp_name = f"robertaIr{args.roberta_lr}_elseIr{args.else_lr}_hiddenDim{args.hidden_dim}_speakerStateDim{args.speaker_state_dim}_timeDim{args.time_dim}_head{args.heads}_localWindowNum{args.local_window_num}_dropout{args.dropout_rate}_Complete"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        speaker_state_dim=args.speaker_state_dim,
        time_dim=args.time_dim,
        heads=args.heads,
        local_window_num=args.local_window_num,
        dropout_rate=args.dropout_rate
    )
    
    # TensorBoard Writer設定
    os.makedirs(f"logs/train/{args.dataset}", exist_ok=True)
    log_dir = os.path.join("runs", "train", args.dataset, exp_name, f"seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    tqdm.write(f"TensorBoard logs will be saved to: {log_dir}")
    
    # モデル全体をGPUに移動 
    model = model.to(device)

    scaler = GradScaler()
    encoder_param_ids = set(id(p) for p in model.text_encoder.parameters())
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": float(args.roberta_lr)},   # RoBERTa 部分
        {"params": other_params, "lr": float(args.else_lr)}                          # それ以外
    ], betas=(0.9, 0.999), weight_decay=5e-3)
    # if (args.time_dim > 0):
    #     time_threshold_param = model.time_threshold
    #     other_params = [
    #         p for p in model.parameters()
    #         if id(p) not in encoder_param_ids and p is not time_threshold_param and p is not None
    #     ]
    #     optimizer = torch.optim.AdamW([
    #         {"params": model.text_encoder.parameters(), "lr": float(args.roberta_lr)},   # RoBERTa 部分
    #         {"params": time_threshold_param, "lr": float(1e-4)},                         # time_threshold
    #         {"params": other_params, "lr": float(args.else_lr)}                          # それ以外
    #     ], betas=(0.9, 0.999), weight_decay=5e-3)
    # else:
    #     other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    #     optimizer = torch.optim.AdamW([
    #         {"params": model.text_encoder.parameters(), "lr": float(args.roberta_lr)},   # RoBERTa 部分
    #         {"params": other_params, "lr": float(args.else_lr)}                          # それ以外
    #     ], betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # データセットとデータローダーの準備
    train_dataset = Dataset(dataset=args.dataset, split='train')
    val_dataset = Dataset(dataset=args.dataset, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CollateFn())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollateFn())

    # ignore以外のdata数を確認
    train_label_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    val_label_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for conv in train_dataset:
        for utt in conv:
            lbl = utt["label"]
            if (lbl != -100):
                train_label_num[lbl] += 1
    for conv in val_dataset:
        for utt in conv:
            lbl = utt["label"]
            if (lbl != -100):
                val_label_num[lbl] += 1
    print("Training set label distribution:", train_label_num)
    print("Validation set label distribution:", val_label_num)


    W = 10   # back-propagation を行うステップ数
    best_f1 = 0.0
    class_counts = torch.tensor([392, 739, 1167, 711, 620, 1149], dtype=torch.float32)
    criterion = make_weighted_ce(class_counts, device)

    for epoch in tqdm(range(args.epochs)):
        # ===== Training =====
        model.train()
        task_loss_lst = []

        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            time_mask = batch["time_mask"].to(device)           # (B, T_max)
            utt_mask = batch["utt_mask"].to(device)
            speeds = batch["speeds"].to(device)
            pauses = batch["pauses"].to(device)
            speakers = batch["speakers"].to(device)
            labels = batch["labels"].to(device)

            _, T_max, _ = input_ids.size()
            train_cache = None
            window_loss_lst = []   # Wステップごとのlossをためるリスト

            for t in range(T_max):
                input_ids_t = input_ids[:, t, :].contiguous()   # (B, U_max)
                utt_mask_t = utt_mask[:, t, :].contiguous()     # (B, U_max)
                speed_t = speeds[:, t].contiguous()             # (B)
                if (t + 1 < T_max):
                    pause_t = pauses[:, t+1].contiguous()       # (B)
                else:
                    pause_t = None                              # ダミー
                label_t = labels[:, t].contiguous()             # (B)

                with autocast():
                    logits, global_x = model(t, input_ids_t, time_mask, utt_mask_t, speed_t, pause_t, train_cache, speakers)
                
                # cache更新
                if (train_cache is None):
                    train_cache = global_x.unsqueeze(1)                                    # (B, 1, hidden)
                else:
                    train_cache = torch.cat([train_cache, global_x.unsqueeze(1)], dim=1)   # (B, t+1, hidden)

                # All labels are -100, skipping loss computation for this timestep.
                if (torch.all(label_t == -100)):
                    pass
                else:
                    with autocast(enabled=False):
                        loss = criterion(logits.float(), label_t)
                    window_loss_lst.append(loss)
                    task_loss_lst.append(loss.detach().item())

                # 勾配更新（10ステップごと or 最後）
                if ((t + 1) % W == 0) or (t + 1 == T_max):
                    if (window_loss_lst):   # 空でないときのみ
                        window_loss = torch.stack(window_loss_lst).mean()
                        scaler.scale(window_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    train_cache = train_cache.detach()
                    model.detach_state()
                    window_loss_lst = []

        scheduler.step()
        tqdm.write(f"epoch_task_loss: {sum(task_loss_lst)}, num_batches: {len(task_loss_lst)}") 
        if (len(task_loss_lst) > 0):
            epoch_task_loss = sum(task_loss_lst) / len(task_loss_lst)
        else:
            epoch_task_loss = 0.0
    
        writer.add_scalars('Loss/Train/Epoch/task_Losses', {'Task': epoch_task_loss}, epoch)
        # writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        tqdm.write(f"Epoch {epoch}, CrossEntropy_loss: {epoch_task_loss:.6f}")


        # ===== Validation =====
        model.eval()
        pred = []
        true = []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):
                input_ids = batch["input_ids"].to(device)
                time_mask = batch["time_mask"].to(device)           # (B, T_max)
                utt_mask = batch["utt_mask"].to(device)
                speeds = batch["speeds"].to(device)
                pauses = batch["pauses"].to(device)
                speakers = batch["speakers"].to(device)
                labels = batch["labels"].to(device)

                _, T_max, _ = input_ids.size()
                val_cache = None

                for t in range(T_max):
                    input_ids_t = input_ids[:, t, :].contiguous()   # (B, U_max)
                    utt_mask_t = utt_mask[:, t, :].contiguous()     # (B, U_max)
                    speed_t = speeds[:, t].contiguous()             # (B)
                    if (t + 1 < T_max):
                        pause_t = pauses[:, t+1].contiguous()       # (B)
                    else:
                        pause_t = None                              # ダミー
                    label_t = labels[:, t].contiguous()             # (B)

                    with autocast():
                        logits, global_x = model(t, input_ids_t, time_mask, utt_mask_t, speed_t, pause_t, val_cache, speakers)

                    # cache更新
                    if (val_cache is None):
                        val_cache = global_x.unsqueeze(1)                                  # (B, 1, hidden)
                    else:
                        val_cache = torch.cat([val_cache, global_x.unsqueeze(1)], dim=1)   # (B, t+1, hidden)

                    # -100をignore
                    valid_idx = (label_t != -100)
                    if (valid_idx.any()):
                        pred_labels = torch.argmax(logits[valid_idx], dim=1).cpu().tolist()
                        true_labels = label_t[valid_idx].cpu().tolist()
                        pred.extend(pred_labels)
                        true.extend(true_labels)

        # F1計算
        metrics = calculate_metrics(pred, true)
        macro_f1 = metrics['f1_macro']
        weighted_f1 = metrics['f1_weighted']
        tqdm.write(f"Macro F1 Score: {macro_f1:.4f}")
        tqdm.write(f"Weighted F1 Score: {weighted_f1:.4f}") 
        writer.add_scalar('F1/Train/Epoch', macro_f1, epoch)

        # クラスごとの詳細結果表示
        # for i in range(args.num_classes):
        #     tqdm.write(f"{metrics[f'correct_num_class_{i}']} / {metrics[f'total_num_class_{i}']}  =>  f1: {metrics[f'f1_class_{i}']:.4f}")

        # 学習可能パラメーター表示
        # if (args.time_dim > 0):
        #     tqdm.write(f"Time thrould: {model.time_threshold.item():.4f}")
            
        # モデル保存
        if (macro_f1 >= best_f1):
            best_f1 = macro_f1
            os.makedirs(
                f"saved_models/{args.dataset}/"
                f"best_{exp_name}/", exist_ok=True
            )
            best_model_path = (
                f"saved_models/{args.dataset}/"
                f"best_{exp_name}/seed{args.seed}.pth"
            )

            # 両方に保存（best_model_path* は毎回上書きされる）
            torch.save(model.state_dict(), best_model_path)

            tqdm.write(f"We've saved the new model (Macro F1 Score: {macro_f1:.4f})")
            tqdm.write(f"Best model (overwritten): {best_model_path}")
        tqdm.write("----------------------------------------------------------------------------")

    tqdm.write(f"Best Macro F1 Score: {best_f1:.4f}")
    writer.close()
    return



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        tqdm.write(f"{arg}: {getattr(_args, arg)}")
    set_seed(_args.seed)
    train(_args)