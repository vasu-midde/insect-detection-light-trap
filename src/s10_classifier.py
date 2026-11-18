"""s10_classifier.py - species classifier on the sorted insect crops (stage 2 of the pipeline).

Input (made with s09_crops.py on the laptop, then copied to the lab PC):
    <base>/species_crops/labels.csv          crop path, site, frame, track, species
    <base>/species_crops/crops/<site>/*.jpg
    <base>/work/<site>/split.json            time-block split made by s03 for the detector

What it does
  1. Merges species with fewer than MIN_TRACKS moths (tracks) into MERGE_INTO and prints the mapping.
  2. Gives every crop the split of its frame (train / val / test), exactly as the detector.
     A moth (track) that sits in two splits keeps only its train crops; its val/test crops are
     dropped, so the same individual is never in train and test.
  3. Fine-tunes an ImageNet-pretrained ResNet-18 (224 x 224), class-weighted loss,
     keeps the epoch with the best val macro-F1, scores test once.
  4. Writes <base>/work/classifier/: classifier_best.pt, summary.md, per_class.csv,
     confusion_test.png, test_predictions.csv, history.png

Run in the notebook:
    import s10_classifier
    s10_classifier.main(base=BASE)
"""
import csv, json, math, os, random, re, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SITES = ["Station_1st_night", "Station_2nd_night", "OAK"]
CFG = dict(min_tracks=8, merge_into="other insect", img=224, batch=32, epochs=25, lr=3e-4,
           weight_decay=1e-4, workers=2, seed=0)


def frame_no(name):
    m = re.search(r"_(\d{4})\.jpe?g$", name, re.I)
    return int(m.group(1)) if m else -1


def load_rows(base, cfg):
    sc = os.path.join(base, "species_crops")
    rows = list(csv.DictReader(open(os.path.join(sc, "labels.csv"), encoding="utf-8")))
    # ---- merge rare species (counted in moths = tracks, not crops)
    tracks = defaultdict(set)
    for r in rows:
        tracks[r["species"]].add((r["site"], r["track"]))
    keep = {s for s, t in tracks.items() if len(t) >= cfg["min_tracks"]}
    keep.add(cfg["merge_into"])
    merged = sorted(s for s in tracks if s not in keep)
    for r in rows:
        r["label"] = r["species"] if r["species"] in keep else cfg["merge_into"]
    classes = sorted(keep - {cfg["merge_into"]}) + [cfg["merge_into"]]
    # ---- split of each crop = split of its frame (detector's time blocks)
    role_of = {}
    for site in SITES:
        p = os.path.join(base, "work", site, "split.json")
        if not os.path.exists(p):
            raise SystemExit("missing %s - run the detector notebook for %s first (step 3)" % (p, site))
        for role, names in json.load(open(p)).items():
            for n in names:
                role_of[(site, frame_no(n))] = role if role in ("train", "val", "test") else "gap"
    for r in rows:
        r["role"] = role_of.get((r["site"], int(r["frame"])), "gap")
        r["path"] = os.path.join(sc, r["crop"].replace("\\", "/"))
    # ---- a moth in two splits: keep its train crops only
    roles_of_track = defaultdict(set)
    for r in rows:
        if r["role"] != "gap":
            roles_of_track[(r["site"], r["track"])].add(r["role"])
    dropped = 0
    for r in rows:
        if r["role"] in ("val", "test") and len(roles_of_track[(r["site"], r["track"])]) > 1:
            r["role"] = "dropped"; dropped += 1
    split = {k: [r for r in rows if r["role"] == k] for k in ("train", "val", "test")}
    return split, classes, merged, dropped, rows


class Crops(Dataset):
    def __init__(self, rows, classes, tf):
        self.rows, self.idx, self.tf = rows, {c: i for i, c in enumerate(classes)}, tf
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), self.idx[r["label"]]


class Rot90:
    """Moths sit on the sheet in any orientation."""
    def __call__(self, im):
        return im.rotate(90 * random.randint(0, 3), expand=True)


def transforms_for(img):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose([Rot90(), transforms.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                transforms.ColorJitter(0.3, 0.3, 0.2, 0.02), transforms.ToTensor(), norm])
    test = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(), norm])
    return train, test


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys, ps, probs = [], [], []
    for x, y in loader:
        out = torch.softmax(model(x.to(device)), 1).cpu()
        ys += y.tolist(); ps += out.argmax(1).tolist(); probs += out.max(1).values.tolist()
    return np.array(ys), np.array(ps), np.array(probs)


def scores(y, p, n):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    prec = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return dict(acc=float(cm.diagonal().sum() / max(cm.sum(), 1)), macro_f1=float(f1[present].mean()) if present.any() else 0.0,
                prec=prec, rec=rec, f1=f1, support=cm.sum(1), cm=cm)


def main(base, **over):
    cfg = dict(CFG, **over)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    out = os.path.join(base, "work", "classifier"); os.makedirs(out, exist_ok=True)
    split, classes, merged, dropped, rows = load_rows(base, cfg)
    n = len(classes)

    print("classes (%d):" % n, classes)
    if merged:
        print("merged into '%s' (fewer than %d moths):" % (cfg["merge_into"], cfg["min_tracks"]), merged)
    print("crops: train %d | val %d | test %d | dropped %d (moth in two splits) | gap %d" % (
        len(split["train"]), len(split["val"]), len(split["test"]), dropped, sum(r["role"] == "gap" for r in rows)))
    print("%-28s %6s %6s %6s" % ("class", "train", "val", "test"))
    for c in classes:
        print("%-28s %6d %6d %6d" % (c, *[sum(r["label"] == c for r in split[k]) for k in ("train", "val", "test")]))
    thin = [c for c in classes if not any(r["label"] == c for r in split["test"])]
    if thin:
        print("NOTE: no test crops for", thin, "- their test scores are not defined; consider a higher min_tracks")

    tf_train, tf_test = transforms_for(cfg["img"])
    workers = 0 if os.name == "nt" else cfg["workers"]
    dl = {k: DataLoader(Crops(split[k], classes, tf_train if k == "train" else tf_test), batch_size=cfg["batch"],
                        shuffle=(k == "train"), num_workers=workers) for k in split}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n)
    model.to(device)
    cnt = Counter(r["label"] for r in split["train"])
    w = torch.tensor([1.0 / math.sqrt(max(cnt[c], 1)) for c in classes]); w = w / w.mean()
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"] * len(dl["train"]))

    best, hist = -1, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time(); losses = []
        for x, y in dl["train"]:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
        yv, pv, _ = predict(model, dl["val"], device)
        sv = scores(yv, pv, n)
        hist.append((ep, float(np.mean(losses)), sv["acc"], sv["macro_f1"]))
        mark = ""
        if sv["macro_f1"] > best:
            best = sv["macro_f1"]; mark = "  <- best"
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": ep, "cfg": cfg}, os.path.join(out, "classifier_best.pt"))
        print("  epoch %2d  loss %.3f  val acc %.3f  val macro-F1 %.3f  %.0fs%s" % (ep, hist[-1][1], sv["acc"], sv["macro_f1"], time.time() - t0, mark))

    # ---- test once, with the best epoch
    ck = torch.load(os.path.join(out, "classifier_best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    yt, pt, prt = predict(model, dl["test"], device)
    st = scores(yt, pt, n)
    print("\nTEST (best epoch %d): accuracy %.3f  macro-F1 %.3f  (%d crops)" % (ck["epoch"], st["acc"], st["macro_f1"], len(yt)))

    md = ["| Class | Test crops | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    with open(os.path.join(out, "per_class.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["class", "test_crops", "precision", "recall", "f1"])
        for i, c in enumerate(classes):
            wr.writerow([c, int(st["support"][i]), round(st["prec"][i], 3), round(st["rec"][i], 3), round(st["f1"][i], 3)])
            md.append("| %s | %d | %.2f | %.2f | %.2f |" % (c, st["support"][i], st["prec"][i], st["rec"][i], st["f1"][i]))
    head = ["# Species classifier (ResNet-18, ImageNet-pretrained)", "",
            "Best epoch %d of %d (val macro-F1 %.3f). Test: accuracy **%.3f**, macro-F1 **%.3f** on %d crops." % (
                ck["epoch"], cfg["epochs"], best, st["acc"], st["macro_f1"], len(yt)),
            "Crops train / val / test: %d / %d / %d; %d val/test crops dropped (moth also in another split)." % (
                len(split["train"]), len(split["val"]), len(split["test"]), dropped),
            "Merged into '%s' (< %d moths): %s" % (cfg["merge_into"], cfg["min_tracks"], ", ".join(merged) or "none"), ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(head + md) + "\n")
    json.dump(dict(classes=classes, merged=merged, test_acc=st["acc"], test_macro_f1=st["macro_f1"], best_epoch=ck["epoch"],
                   confusion=st["cm"].tolist()), open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(out, "test_predictions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["crop", "site", "frame", "track", "true", "pred", "confidence"])
        for r, a, b, c in zip(split["test"], yt, pt, prt):
            wr.writerow([r["crop"], r["site"], r["frame"], r["track"], classes[a], classes[b], round(float(c), 3)])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm = st["cm"]; rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.0 * n + 1.5))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if rown[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix (counts; colour = row share)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_test.png"), dpi=150); plt.close(fig)
    h = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h[:, 0], h[:, 1], "o-"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(h[:, 0], h[:, 2], "o-", label="val accuracy"); ax[1].plot(h[:, 0], h[:, 3], "o-", label="val macro-F1")
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(os.path.join(out, "history.png"), dpi=150); plt.close(fig)
    print("results ->", out)
    return st
