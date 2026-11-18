"""Step 8 - look at what went wrong on the test frames (at the threshold chosen on val).

Every false alarm and every miss is cropped and sorted by cause:
  false alarm : localization (overlaps an insect, IoU 0.1-0.5) | duplicate (insect already found)
                | background (nothing there: stain, mesh, info bar)
  miss        : small (< 100 px) | touching (overlaps another insect) | on info bar | other
Writes results/errors_<mode>/<cause>.jpg contact sheets, errors_<mode>.csv, and
results/examples_<mode>/ with whole test frames (green = hand label, red = model).
"""
import csv
import shutil

import cv2
import numpy as np

from common import area, coco_to_boxes, iou, list_frames, load_json, match, read_rgb
from config import CFG, paths
from s06_infer import load_preds


def sheet(crops, path, cols=8, size=160):
    if not crops:
        return
    tiles = []
    for c, txt in crops[:cols * 6]:
        t = cv2.resize(c, (size, size))
        cv2.putText(t, txt, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    g = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))


def crop(img, b, ctx=0.6):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    s = max(w, h) * (1 + 2 * ctx) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = img.shape[:2]
    cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
    X0, Y0, X1, Y1 = [int(v) for v in (max(cx - s, 0), max(cy - s, 0), min(cx + s, W), min(cy + s, H))]
    X1, Y1 = max(X1, X0 + 2), max(Y1, Y0 + 2)
    c = np.ascontiguousarray(img[Y0:Y1, X0:X1])
    cv2.rectangle(c, (int(x0 - X0), int(y0 - Y0)), (int(x1 - X0), int(y1 - Y0)), (255, 0, 0), max(2, int(s / 60)))
    return c


def main(cfg=CFG, mode="tiled", n_examples=6):
    p = paths(cfg)
    gts, _ = coco_to_boxes(load_json(p["labels"]))
    split = load_json(p["split"])
    preds, _ = load_preds(p["preds"] / ("%s_test.json" % mode))
    thr = load_json(p["results"] / "summary.json")[mode]["threshold_from_val"]
    raw = {f.name: f for f in list_frames(cfg["raw_dir"])}
    bar_y = cfg["frame_size"][1] * (1 - cfg["info_bar_frac"])
    out = p["results"] / ("errors_%s" % mode)
    ex = p["results"] / ("examples_%s" % mode)
    for d in (out, ex):                       # clear old runs so stale crops never mix in
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cats = {k: [] for k in ("fp_localization", "fp_duplicate", "fp_background",
                            "fn_small", "fn_touching", "fn_info_bar", "fn_other")}
    rows = []
    test = split["test"]
    # examples: the frames with the most insects, plus the frames with the largest count error
    by_err = sorted(test, key=lambda n: -abs(int((preds[n][1] >= thr).sum()) - len(gts[n])))
    by_n = sorted(test, key=lambda n: -len(gts[n]))
    show = list(dict.fromkeys(by_n[:n_examples // 2] + by_err))[:n_examples]

    for name in test:
        g = gts[name]
        b, s = preds[name]
        m = s >= thr
        b, s = b[m], s[m]
        order, tp, used = match(g, b, s, cfg["eval_iou"])
        img = None
        if (~tp).any() or (~used).any() or name in show:
            img = read_rgb(raw[name])
        bs = b[order]
        ious = iou(bs, g) if len(g) and len(bs) else np.zeros((len(bs), len(g)))
        for k in np.where(~tp)[0]:
            best = ious[k].max() if len(g) else 0
            c = "fp_duplicate" if best >= cfg["eval_iou"] else "fp_localization" if best >= 0.1 else "fp_background"
            cats[c].append((crop(img, bs[k]), "%s s%.2f" % (name[-8:-4], s[order][k])))
            rows.append([name, c] + [round(float(v)) for v in bs[k]] + [round(float(s[order][k]), 3)])
        gg = iou(g, g) if len(g) else np.zeros((0, 0))
        if len(g):
            np.fill_diagonal(gg, 0)
        for j in np.where(~used)[0]:
            side = float(np.sqrt(area(g[j:j + 1]))[0])
            c = ("fn_small" if side < 100 else "fn_touching" if gg[j].max() > 0 else
                 "fn_info_bar" if g[j, 3] > bar_y else "fn_other")
            cats[c].append((crop(img, g[j]), "%s %dpx" % (name[-8:-4], side)))
            rows.append([name, c] + [round(float(v)) for v in g[j]] + [""])
        if name in show:
            vis = img.copy()
            for x0, y0, x1, y1 in g.astype(int):
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 10)
            for (x0, y0, x1, y1), sc in zip(b.astype(int), s):
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 6)
                cv2.putText(vis, "%.2f" % sc, (x0, max(y0 - 12, 40)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 0, 0), 5)
            vis = cv2.resize(vis, (vis.shape[1] // 4, vis.shape[0] // 4), interpolation=cv2.INTER_AREA)
            cv2.putText(vis, "%s  true %d  model %d  (green = hand label, red = model)" % (name, len(g), len(b)),
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.imwrite(str(ex / name.replace(".JPG", ".jpg")), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    for c, lst in cats.items():
        sheet(lst, out / ("%s.jpg" % c))
    with open(p["results"] / ("errors_%s.csv" % mode), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "cause", "x0", "y0", "x1", "y1", "score"]); w.writerows(rows)
    summary = {c: len(v) for c, v in cats.items()}
    print("[%s] errors on test at thr %.2f:" % (mode, thr), summary)
    print("contact sheets -> %s   examples -> %s" % (out, ex))
    return summary


if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "tiled")
