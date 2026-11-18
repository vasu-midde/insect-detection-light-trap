"""Step 7 - the numbers for the slides.

For each model (tiled, down):
  1. pick the counting threshold on VAL (lowest count MAE)  - test is never used for any choice
  2. report on TEST once: AP50, AP50:95, precision, recall, F1, count MAE, count bias, s/frame,
     plus recall by insect size (small < 100 px, medium 100-300 px, large > 300 px)
Writes results/summary.json, results/summary.md and plots (PR curves, true-vs-predicted counts,
count over the night if predictions for 'all' frames exist).
"""
from pathlib import Path

import numpy as np

from common import (area, average_precision, coco_to_boxes, counts_at, frame_index, frame_time, load_json,
                    map_50_95, match, pick_threshold, save_json)
from config import CFG, paths
from s06_infer import load_preds

SIZE_BINS = [("small <100px", 0, 100), ("medium 100-300px", 100, 300), ("large >300px", 300, 1e9)]


def recall_by_size(gts, preds, thr, iou_thr):
    hit = {k: [0, 0] for k, _, _ in SIZE_BINS}
    for name, g in gts.items():
        if not len(g):
            continue
        b, s = preds.get(name, (np.zeros((0, 4)), np.zeros(0)))
        m = s >= thr
        _, _, used = match(g, b[m], s[m], iou_thr)
        side = np.sqrt(area(g))
        for lab, lo, hi in SIZE_BINS:
            sel = (side >= lo) & (side < hi)
            hit[lab][0] += int(used[sel].sum()); hit[lab][1] += int(sel.sum())
    return {k: {"recall": (v[0] / v[1]) if v[1] else None, "n": v[1]} for k, v in hit.items()}


def evaluate_mode(cfg, p, mode, gts_all, split):
    vf, tf = p["preds"] / ("%s_val.json" % mode), p["preds"] / ("%s_test.json" % mode)
    if not (vf.exists() and tf.exists()):
        return None
    pv, _ = load_preds(vf)
    pt, secs = load_preds(tf)
    gv = {n: gts_all[n] for n in split["val"]}
    gt = {n: gts_all[n] for n in split["test"]}
    thr = pick_threshold(gv, pv, cfg["eval_iou"])["thr"]
    c = counts_at(gt, pt, thr, cfg["eval_iou"])
    ap50, curve = average_precision(gt, pt, 0.5)
    res = {
        "mode": mode, "threshold_from_val": thr,
        "val_count_mae": pick_threshold(gv, pv, cfg["eval_iou"])["count_mae"],
        "test_frames": len(gt), "test_insects": int(sum(len(v) for v in gt.values())),
        "AP50": ap50, "AP50_95": map_50_95(gt, pt),
        "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
        "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        "count_mae": c["count_mae"], "count_bias": c["count_bias"],
        "true_total": int(sum(len(v) for v in gt.values())), "pred_total": int(sum(x[2] for x in c["per_image"])),
        "seconds_per_frame": float(np.mean(list(secs.values()))),
        "recall_by_size": recall_by_size(gt, pt, thr, cfg["eval_iou"]),
        "per_image": c["per_image"],
    }
    return res, curve


def plots(p, results, curves, gts_all, cfg):   # CHANGED: cfg added (site name for the title)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"tiled": "#1f6f8b", "down": "#e07b39"}
    lab = {"tiled": "tiled", "down": "whole frame, downscaled"}

    fig, ax = plt.subplots(figsize=(4.6, 4))
    for m, cu in curves.items():
        if cu is not None:
            ax.plot(cu["recall"], cu["precision"], color=col[m], label="%s (AP50 %.2f)" % (lab[m], results[m]["AP50"]))
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Test set, IoU 0.5"); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(p["results"] / "pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    mx = 1
    for m, r in results.items():
        t = np.array([x[1] for x in r["per_image"]]); pr = np.array([x[2] for x in r["per_image"]])
        mx = max(mx, t.max(initial=0), pr.max(initial=0))
        ax.scatter(t, pr, s=22, alpha=0.75, color=col[m], label="%s (MAE %.2f)" % (lab[m], r["count_mae"]))
    ax.plot([0, mx + 1], [0, mx + 1], "k--", lw=1)
    ax.set_xlabel("true insects per frame"); ax.set_ylabel("predicted insects per frame")
    ax.set_title("Counting on test frames"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(p["results"] / "count_scatter.png", dpi=150); plt.close(fig)

    # count over the night (needs predictions on all good frames)
    for m, r in results.items():
        f = p["preds"] / ("%s_all.json" % m)
        if not f.exists():
            continue
        pa, _ = load_preds(f)
        thr = r["threshold_from_val"]
        names = sorted(pa, key=frame_index)
        x = [frame_time(n)[11:16] for n in names]
        y = [int((pa[n][1] >= thr).sum()) for n in names]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(range(len(y)), y, color=col[m], lw=1.6, label="model count (%s)" % lab[m])
        lab_idx = [names.index(n) for n in gts_all if n in names]
        ax.scatter(lab_idx, [len(gts_all[names[i]]) for i in lab_idx], s=10, color="k", zorder=3, label="hand count (annotated frames)")
        step = max(1, len(x) // 10)
        ax.set_xticks(range(0, len(x), step)); ax.set_xticklabels(x[::step])
        ax.set_ylabel("insects on sheet"); ax.set_title("%s, night starting %s" % (Path(cfg["raw_dir"]).name, frame_time(names[0])[:10])); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(p["results"] / ("count_over_night_%s.png" % m), dpi=150); plt.close(fig)


def main(cfg=CFG):
    p = paths(cfg)
    gts_all, _ = coco_to_boxes(load_json(p["labels"]))
    split = load_json(p["split"])
    results, curves = {}, {}
    for m in ("tiled", "down"):
        r = evaluate_mode(cfg, p, m, gts_all, split)
        if r:
            results[m], curves[m] = r
    if not results:
        raise SystemExit("no predictions found - run s06_infer for val and test first")
    plots(p, results, curves, gts_all, cfg)
    save_json({m: {k: v for k, v in r.items() if k != "per_image"} for m, r in results.items()}, p["results"] / "summary.json")

    rows = [("Test frames / insects", lambda r: "%d / %d" % (r["test_frames"], r["test_insects"])),
            ("Threshold (chosen on val)", lambda r: "%.2f" % r["threshold_from_val"]),
            ("AP50", lambda r: "%.3f" % r["AP50"]), ("AP50:95", lambda r: "%.3f" % r["AP50_95"]),
            ("Precision", lambda r: "%.3f" % r["precision"]), ("Recall", lambda r: "%.3f" % r["recall"]),
            ("F1", lambda r: "%.3f" % r["f1"]),
            ("Count MAE (insects/frame)", lambda r: "%.2f" % r["count_mae"]),
            ("Count bias (pred - true)", lambda r: "%+.2f" % r["count_bias"]),
            ("Total count true / pred", lambda r: "%d / %d" % (r["true_total"], r["pred_total"])),
            ("Seconds per frame", lambda r: "%.2f" % r["seconds_per_frame"])]
    for lab, _, _ in SIZE_BINS:
        rows.append(("Recall, " + lab, lambda r, lab=lab: "-" if r["recall_by_size"][lab]["recall"] is None else
                     "%.2f (n=%d)" % (r["recall_by_size"][lab]["recall"], r["recall_by_size"][lab]["n"])))
    modes = list(results)
    head = {"tiled": "Tiled (1/2 res, %d px tiles)" % cfg["tile"],
            "down": "Whole frame (1/%d res)" % (2 * cfg["down_factor"])}
    md = ["| Metric | " + " | ".join(head[m] for m in modes) + " |", "|---|" + "---|" * len(modes)]
    for lab, fn in rows:
        md.append("| %s | %s |" % (lab, " | ".join(fn(results[m]) for m in modes)))
    text = "\n".join(md)
    (p["results"] / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return results


if __name__ == "__main__":
    main()
