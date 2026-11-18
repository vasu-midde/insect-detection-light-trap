"""Shared helpers: frame I/O, COCO files, box maths, merge, and detection metrics.
Only numpy / opencv here, so it runs anywhere (no torch needed)."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"_(\d{12})_(\d{4})\.jpe?g$", re.I)   # NSCF----_YYMMDDhhmmss_NNNN.JPG


# ----------------------------------------------------------------- frames
def list_frames(raw_dir):
    """All camera frames under raw_dir, sorted by their running number."""
    fs = [p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg") and FRAME_RE.search(p.name)]
    return sorted(fs, key=lambda p: int(FRAME_RE.search(p.name).group(2)))


def frame_index(name):
    return int(FRAME_RE.search(Path(name).name).group(2))


def frame_time(name):
    s = FRAME_RE.search(Path(name).name).group(1)          # YYMMDDhhmmss
    return "20%s-%s-%s %s:%s:%s" % (s[0:2], s[2:4], s[4:6], s[6:8], s[8:10], s[10:12])


def check_jpeg(path):
    """Cheap integrity test: real JPEG header, an end marker near the end of file,
    and the bottom rows decode to something other than flat grey."""
    path = Path(path)
    size = path.stat().st_size
    if size < 100_000:
        return False, "empty/too small (%d bytes)" % size
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(max(0, size - 4096))
        tail = f.read()
    if head[:3] != b"\xff\xd8\xff":
        return False, "not a JPEG (bad header)"
    if b"\xff\xd9" not in tail:
        return False, "truncated (no end marker)"
    return True, "ok"


def read_rgb(path, reduce=1):
    """Read a frame as RGB uint8. reduce in {1,2,4,8} uses libjpeg's fast DCT downscaling."""
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
    im = cv2.imread(str(path), flag)
    if im is None:
        raise IOError("cannot decode %s" % path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- COCO
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def coco_to_boxes(coco):
    """-> {file_name: np.array (N,4) xyxy}, plus image size dict. Every image appears, even with 0 boxes."""
    by_id = {im["id"]: im for im in coco["images"]}
    out = {Path(im["file_name"]).name: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        out[Path(by_id[a["image_id"]]["file_name"]).name].append([x, y, x + w, y + h])
    sizes = {Path(im["file_name"]).name: (im["width"], im["height"]) for im in coco["images"]}
    return {k: np.array(v, dtype=np.float32).reshape(-1, 4) for k, v in out.items()}, sizes


def boxes_to_coco(boxes_by_name, sizes, scores_by_name=None, category="insect"):
    images, anns = [], []
    aid = 1
    for i, name in enumerate(sorted(boxes_by_name), start=1):
        w, h = sizes[name]
        images.append({"id": i, "file_name": name, "width": int(w), "height": int(h)})
        for j, b in enumerate(boxes_by_name[name]):
            x0, y0, x1, y1 = [float(v) for v in b]
            a = {"id": aid, "image_id": i, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                 "area": (x1 - x0) * (y1 - y0), "iscrowd": 0, "segmentation": [], "attributes": {"occluded": False}}
            if scores_by_name is not None:
                a["score"] = float(scores_by_name[name][j])
            anns.append(a)
            aid += 1
    return {"info": {"description": "Oak insects"}, "licenses": [],
            "categories": [{"id": 1, "name": category, "supercategory": ""}],
            "images": images, "annotations": anns}


# ----------------------------------------------------------------- boxes
def area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def inter(a, b):
    """pairwise intersection area, a (N,4) b (M,4) -> (N,M)"""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def iou(a, b):
    i = inter(a, b)
    return i / (area(a)[:, None] + area(b)[None, :] - i + 1e-9)


def merge_boxes(boxes, scores, iou_thr=0.5, contain_thr=0.7):
    """Greedy merge for tiled detections. A lower-scoring box is dropped when it overlaps a kept box
    with IoU >= iou_thr, OR when >= contain_thr of its own area lies inside the kept box
    (that is the piece of an insect cut off at a tile edge; plain NMS keeps it and double-counts).
    Memory is O(N): one row of overlaps at a time."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    boxes = boxes[order]
    ar = area(boxes)
    alive = np.ones(len(boxes), bool)
    keep = []
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.where(alive[i + 1:])[0] + i + 1
        if len(rest) == 0:
            break
        it = inter(boxes[i:i + 1], boxes[rest])[0]
        u = ar[i] + ar[rest] - it
        dup = (it / (u + 1e-9) >= iou_thr) | (it / (np.minimum(ar[i], ar[rest]) + 1e-9) >= contain_thr)
        alive[rest[dup]] = False
    return order[np.array(keep)]


def tile_grid(w, h, tile, overlap):
    """Top-left corners of tiles covering a w x h image; last row/col is snapped to the border."""
    stride = tile - overlap
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


# ----------------------------------------------------------------- metrics
def match(gt, pred, scores, iou_thr):
    """Greedy match by score. Returns (order, tp, gt_used): pred order by score, TP flag per sorted pred,
    and which ground-truth boxes were found."""
    order = np.argsort(-scores)
    pred = pred[order]
    tp = np.zeros(len(pred), bool)
    gt_used = np.zeros(len(gt), bool)
    if len(gt) and len(pred):
        ious = iou(pred, gt)
        for k in range(len(pred)):
            cand = np.where(~gt_used & (ious[k] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                gt_used[j] = True
                tp[k] = True
    return order, tp, gt_used


def average_precision(gts, preds, iou_thr=0.5):
    """COCO-style 101-point AP over a set of images. gts: {name: (N,4)}, preds: {name: (boxes, scores)}"""
    all_s, all_tp, n_gt = [], [], 0
    for name, g in gts.items():
        b, s = preds.get(name, (np.zeros((0, 4)), np.zeros(0)))
        n_gt += len(g)
        order, tp, _ = match(g, b, s, iou_thr)
        all_s.append(s[order]); all_tp.append(tp)
    if n_gt == 0:
        return float("nan"), None
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    o = np.argsort(-s, kind="stable")
    tp = tp[o]; s = s[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1)
    # precision envelope
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    rs = np.linspace(0, 1, 101)
    ap = np.mean([prec[np.searchsorted(rec, r, side="left")] if np.any(rec >= r) else 0.0 for r in rs])
    return float(ap), {"recall": rec, "precision": prec, "scores": s}


def map_50_95(gts, preds):
    aps = [average_precision(gts, preds, t)[0] for t in np.arange(0.5, 0.96, 0.05)]
    return float(np.mean(aps))


def counts_at(gts, preds, thr, iou_thr=0.5):
    """Precision / recall / F1 / count errors at one score threshold."""
    tp = fp = fn = 0
    errs = []
    rows = []
    for name, g in gts.items():
        b, s = preds.get(name, (np.zeros((0, 4)), np.zeros(0)))
        m = s >= thr
        order, t, _ = match(g, b[m], s[m], iou_thr)
        tp += int(t.sum()); fp += int((~t).sum()); fn += int(len(g) - t.sum())
        errs.append(int(m.sum()) - len(g))
        rows.append((name, len(g), int(m.sum())))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    errs = np.array(errs)
    return {"thr": float(thr), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9), "count_mae": float(np.abs(errs).mean()) if len(errs) else float("nan"),
            "count_bias": float(errs.mean()) if len(errs) else float("nan"), "per_image": rows}


def pick_threshold(gts, preds, iou_thr=0.5):
    """Score threshold with the lowest count MAE on this set (ties -> higher F1)."""
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        c = counts_at(gts, preds, t, iou_thr)
        key = (c["count_mae"], -c["f1"])
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]
