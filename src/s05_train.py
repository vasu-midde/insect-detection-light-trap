"""Step 5 - train Faster R-CNN, either on tiles (mode="tiled") or on downscaled frames (mode="down").

Augmentation: horizontal + vertical flip, 90-degree rotation (tiles only, they are square) and mild
brightness/contrast jitter (lighting changes over the night). Boxes are moved with the pixels.
Model selection: best AP50 on the val tiles/frames after each epoch -> models/<mode>_best.pt
"""
import csv
import math
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from common import average_precision, load_json
from config import CFG, paths
from model import build_model


class BoxDataset(Dataset):
    def __init__(self, items, train, square):
        self.items, self.train, self.square = items, train, square

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.cvtColor(cv2.imread(it["file"]), cv2.COLOR_BGR2RGB)
        b = np.array(it["boxes"], np.float32).reshape(-1, 4)
        if self.train:
            img, b = augment(img, b, self.square)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.from_numpy(b).float(), "labels": torch.ones(len(b), dtype=torch.int64)}
        return t, target


def augment(img, b, square):
    h, w = img.shape[:2]
    if random.random() < 0.5:                                   # horizontal flip
        img = img[:, ::-1]; b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
    if random.random() < 0.5:                                   # vertical flip
        img = img[::-1]; b = b.copy(); b[:, [1, 3]] = h - b[:, [3, 1]]
    if square and random.random() < 0.5:                        # rotate 90 deg clockwise
        img = np.rot90(img, k=-1)                               # (x, y) -> (h - y, x)
        b = np.stack([h - b[:, 3], b[:, 0], h - b[:, 1], b[:, 2]], 1) if len(b) else b
    if random.random() < 0.8:                                   # lighting
        a = random.uniform(0.8, 1.2); c = random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * a + c, 0, 255).astype(np.uint8)
    return img, b.astype(np.float32)


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate_items(model, items, device, batch=2):
    """AP50 on a list of {file, boxes} images (tiles or downscaled frames)."""
    model.eval()
    gts, preds = {}, {}
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        imgs = [torch.from_numpy(cv2.cvtColor(cv2.imread(it["file"]), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
                for it in chunk]
        out = model(imgs)
        for it, o in zip(chunk, out):
            gts[it["file"]] = np.array(it["boxes"], np.float32).reshape(-1, 4)
            preds[it["file"]] = (o["boxes"].cpu().numpy(), o["scores"].cpu().numpy())
    return average_precision(gts, preds, 0.5)[0]


def main(cfg=CFG, mode="tiled", pretrained=True, max_iters=None):
    p = paths(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = p["tiles"] if mode == "tiled" else p["down"]
    train_items, val_items = load_json(src / "train.json"), load_json(src / "val.json")
    if len(val_items) > cfg["val_max_images"]:          # fixed subset keeps the per-epoch check quick
        val_items = random.Random(0).sample(val_items, cfg["val_max_images"])
    print("[%s] %d train / %d val images on %s" % (mode, len(train_items), len(val_items), device))

    bs = cfg["batch_size"] if mode == "tiled" else 1
    ds = BoxDataset(train_items, True, mode == "tiled")
    n_per_epoch = min(len(ds), cfg["iters_per_epoch"] * bs)
    dl = DataLoader(ds, batch_size=bs, sampler=RandomSampler(ds, num_samples=n_per_epoch),
                    num_workers=cfg["num_workers"], collate_fn=collate, drop_last=n_per_epoch > bs)
    model = build_model(cfg, mode, pretrained).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    steps = epochs * len(dl)
    warm = min(len(dl), 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist_path = p["models"] / ("%s_history.csv" % mode)
    best, it = -1.0, 0
    mm = cfg["max_minutes"][mode] if isinstance(cfg["max_minutes"], dict) else cfg["max_minutes"]
    budget = float("inf") if mm is None else mm * 60          # CHANGED: None = no time cap
    t_start = time.time()
    with open(hist_path, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["epoch", "loss", "val_ap50", "seconds", "lr"])
        for ep in range(1, epochs + 1):
            model.train(); t0 = time.time(); losses = []
            for imgs, tg in dl:
                imgs = [x.to(device) for x in imgs]
                tg = [{k: v.to(device) for k, v in t.items()} for t in tg]
                try:
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss = sum(model(imgs, tg).values())
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
                    scaler.step(opt); scaler.update(); sched.step()
                    losses.append(loss.item())
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print("  out of GPU memory on one batch - skipped (lower batch_size if this repeats)")
                it += 1
                if max_iters and it >= max_iters:
                    break
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ap = evaluate_items(model, val_items, device)
            dt = time.time() - t0
            lr = opt.param_groups[0]["lr"]
            wr.writerow([ep, np.mean(losses) if losses else float("nan"), ap, round(dt, 1), lr]); fh.flush()
            mark = ""
            if ap > best or best < 0:
                best = ap
                torch.save({"model": model.state_dict(), "epoch": ep, "val_ap50": ap, "mode": mode}, p["models"] / ("%s_best.pt" % mode))
                mark = "  <- best"
            print("  epoch %2d  loss %.3f  val AP50 %.3f  %.0fs%s" % (ep, np.mean(losses) if losses else float("nan"), ap, dt, mark))
            if ep == 1:
                print("  (about %.0f min for all %d epochs at this speed; time cap %s)" % (dt * epochs / 60, epochs, "none" if mm is None else "%.0f min" % mm))  # CHANGED
            if max_iters and it >= max_iters:
                break
            if time.time() - t_start + dt > budget:
                print("  time cap reached - stopping after epoch %d (best epoch is kept)" % ep)
                break
    torch.save({"model": model.state_dict(), "epoch": ep, "mode": mode}, p["models"] / ("%s_last.pt" % mode))
    plot_history(hist_path)
    print("best val AP50 %.3f -> %s" % (best, p["models"] / ("%s_best.pt" % mode)))
    return best


def plot_history(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [float(r["loss"]) for r in rows], marker="o"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [float(r["val_ap50"]) for r in rows], marker="o", color="tab:orange"); ax[1].set_title("val AP50")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(path).replace(".csv", ".png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "tiled")
