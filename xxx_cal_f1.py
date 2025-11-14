import re, statistics
import os, glob


# ==== 読み込むログファイルをここで並べる ====
dir_name = "robertaIr2e-5_elseIr1e-4_hiddenDim384_emotionDim64_pauseDim0_head6_localWindowNum0_dropout0.1_NoEmotionVec"

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
    overall_m = re.search(r"F1 Score:\s*([0-9.+-eE]+)", txt)
    if (not overall_m):
        raise ValueError(f"Overall F1 not found in {path}")
    overall = float(overall_m.group(1))

    class_f1 = [float(x) for x in re.findall(r"f1:\s*([0-9.+-eE]+)", txt)]

    thr_m = re.search(r"Learned time threshold:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", txt)
    threshold = float(thr_m.group(1)) if thr_m else None

    return overall, class_f1, threshold


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    return f"{m:.4f} ± {s:.4f}"


overall_all, per_cls, thresholds = [], [], []

for f in files:
    fp = f"./logs/test/IEMOCAP/{f}"
    o, c, thr = parse_log(fp)
    overall_all.append(o)
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
print(f"Overall F1: {mean_std(overall_all)}")
for i, c in enumerate(per_cls):
    print(f"Class {i}: {mean_std(c)}")

if (thresholds):
    print(f"Learned time threshold: {mean_std(thresholds)}")
else:
    print("No 'Learned time threshold' found in any log.")