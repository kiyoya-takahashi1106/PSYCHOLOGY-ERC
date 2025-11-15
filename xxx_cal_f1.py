import re, statistics
import os, glob


# ==== 読み込むログファイルをここで並べる ====
dir_name = "robertaIr2e-5_elseIr1e-4_hiddenDim768_emotionDim64_pauseDim0_head6_localWindowNum0_dropout0.1_BaseLine"

target = os.path.join("logs", "test", "IEMOCAP", dir_name)
if (not os.path.isdir(target)):
    raise FileNotFoundError(f"Directory not found: {target}")
files = sorted(
    f"{dir_name}/{os.path.basename(p)}"
    for p in glob.glob(os.path.join(target, "*"))
    if os.path.isfile(p) and p.lower().endswith((".txt", ".log"))
)
# ================================================


def parse_log(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    
    # Macro F1 Score
    macro_m = re.search(r"Macro F1 Score:\s*([0-9.+-eE]+)", txt)
    if (not macro_m):
        raise ValueError(f"Macro F1 Score not found in {path}")
    macro_f1 = float(macro_m.group(1))
    
    # Weighted F1 Score
    weighted_m = re.search(r"Weighted F1 Score:\s*([0-9.+-eE]+)", txt)
    weighted_f1 = float(weighted_m.group(1)) if weighted_m else None

    # Class-wise F1: "81 / 143  =>  f1: 0.4924" のようなパターン
    class_f1 = [float(x) for x in re.findall(r"=>\s*f1:\s*([0-9.+-eE]+)", txt)]

    # Learned time threshold
    thr_m = re.search(r"Learned time threshold:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", txt)
    threshold = float(thr_m.group(1)) if thr_m else None

    return macro_f1, weighted_f1, class_f1, threshold


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    return f"{m:.4f} ± {s:.4f}"


macro_all, weighted_all, per_cls, thresholds = [], [], [], []

for f in files:
    fp = f"./logs/test/IEMOCAP/{f}"
    macro, weighted, c, thr = parse_log(fp)
    macro_all.append(macro)
    if (weighted is not None):
        weighted_all.append(weighted)
    if (not per_cls):
        per_cls = [[] for _ in c]
    for i, v in enumerate(c):
        per_cls[i].append(v)
    if (thr is not None):
        thresholds.append(thr)

print("===== Summary over runs =====")
print(f"Files ({len(files)}):")
for f in files:
    print(" -", f)
print()
print(f"Macro F1 Score: {mean_std(macro_all)}")
if (weighted_all):
    print(f"Weighted F1 Score: {mean_std(weighted_all)}")
for i, c in enumerate(per_cls):
    print(f"Class {i} F1: {mean_std(c)}")

if (thresholds):
    print(f"Learned time threshold: {mean_std(thresholds)}")