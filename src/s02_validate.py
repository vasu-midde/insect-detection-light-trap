"""Step 2 - check the CVAT export and measure the insects.

Reads cfg["cvat_coco"] (CVAT: Export task dataset -> COCO 1.0), and
  * drops degenerate boxes, clips boxes to the image, reports duplicates (IoU > 0.9 on one frame)
  * warns about frames that are in the pilot set but missing from the export (= not annotated)
  * writes labels_clean.json (what the rest of the pipeline trains and tests on)
  * writes label_stats.json  (box sizes; anchor-size suggestion; per-frame counts)
"""
import numpy as np

from common import boxes_to_coco, coco_to_boxes, iou, load_json, save_json
from config import CFG, paths


def main(cfg=CFG):
    p = paths(cfg)
    coco = load_json(cfg["cvat_coco"])
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    print("categories in export:", cats, "(all are treated as one class 'insect')")
    boxes, sizes = coco_to_boxes(coco)
    # CHANGED: CVAT writes '.jpg' in the export while the files are '.JPG' -> use the names on disk
    from common import list_frames
    disk = {f.name.lower(): f.name for f in list_frames(cfg["raw_dir"])}
    missing_files = [n for n in boxes if n.lower() not in disk]
    if missing_files:
        raise SystemExit("%d annotated frames are not in raw_dir %s, e.g. %s" % (len(missing_files), cfg["raw_dir"], missing_files[:3]))
    boxes = {disk[n.lower()]: v for n, v in boxes.items()}
    sizes = {disk[n.lower()]: v for n, v in sizes.items()}

    issues = []
    clean = {}
    for name, b in boxes.items():
        W, H = sizes[name]
        b = b.copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) if len(b) else np.zeros((0, 2))
        ok = (wh[:, 0] >= 8) & (wh[:, 1] >= 8) if len(b) else np.zeros(0, bool)
        if (~ok).any():
            issues.append("%s: %d tiny/degenerate boxes dropped" % (name, int((~ok).sum())))
        b = b[ok]
        if len(b) > 1:
            m = iou(b, b); np.fill_diagonal(m, 0)
            dup = np.argwhere(np.triu(m) > 0.9)
            if len(dup):
                issues.append("%s: %d near-duplicate box pairs (check in CVAT)" % (name, len(dup)))
                b = np.delete(b, np.unique(dup[:, 1]), axis=0)
        clean[name] = b

    pilot = sorted(x.name for x in p["pilot_dir"].glob("*.JPG")) if p["pilot_dir"].exists() else []
    missing = [n for n in pilot if n not in clean]
    if missing:
        issues.append("%d pilot frames are not in the export: %s ..." % (len(missing), missing[:3]))

    allb = np.concatenate([v for v in clean.values() if len(v)] or [np.zeros((0, 4))])
    n_frames, n_boxes = len(clean), len(allb)
    if n_boxes == 0:
        raise SystemExit("The export has no boxes - did CVAT export the right task / format (COCO 1.0)?")
    w, h = allb[:, 2] - allb[:, 0], allb[:, 3] - allb[:, 1]
    side = np.sqrt(w * h)
    q = np.percentile(side, [5, 25, 50, 75, 95, 100])
    # five FPN levels: spread anchors over the real size range (rounded to multiples of 16)
    lo, hi = max(16, np.percentile(side, 2)), np.percentile(side, 99.5)
    anchors = [int(round(v / 16.0) * 16) for v in np.geomspace(lo, hi, 5)]
    counts = {n: int(len(v)) for n, v in clean.items()}
    stats = {
        "frames": n_frames, "boxes": n_boxes,
        "frames_with_insects": int(sum(c > 0 for c in counts.values())),
        "boxes_per_frame_mean": float(np.mean(list(counts.values()))),
        "boxes_per_frame_max": int(max(counts.values())),
        "box_side_px_percentiles": dict(zip(["p5", "p25", "p50", "p75", "p95", "max"], [float(x) for x in q])),
        "box_w_max": float(w.max()), "box_h_max": float(h.max()),
        "aspect_ratio_p5_p95": [float(x) for x in np.percentile(w / h, [5, 95])],
        "anchor_suggestion": anchors,
        "issues": issues, "per_frame_counts": counts,
    }
    save_json(stats, p["stats"])
    save_json(boxes_to_coco(clean, sizes), p["labels"])

    print("frames: %d (%d with insects)   boxes: %d   mean/frame: %.1f   max/frame: %d" % (
        n_frames, stats["frames_with_insects"], n_boxes, stats["boxes_per_frame_mean"], stats["boxes_per_frame_max"]))
    print("insect size sqrt(w*h) px: p5 %.0f | p50 %.0f | p95 %.0f | max %.0f   (largest side %.0f px)" % (
        q[0], q[2], q[4], q[5], max(w.max(), h.max())))
    print("anchor suggestion:", anchors, " (config has", cfg["anchor_sizes"], ")")
    big = max(w.max(), h.max()) / cfg["tile_scale"]
    if big > cfg["overlap"]:
        print("WARNING: largest insect (%.0f px after 1/%d shrink) is bigger than the tile overlap (%d px) - raise 'overlap'." % (
            big, cfg["tile_scale"], cfg["overlap"]))
    for s in issues:
        print("  !", s)
    return stats


if __name__ == "__main__":
    main()
