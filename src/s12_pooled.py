"""s12_pooled.py - one detector trained on all three sites together (OAK + Station night 1 + night 2).

It reuses what the per-site runs already made (nothing is re-annotated or re-split):
  work/<site>/tiles/{train,val}.json, work/<site>/downscaled/{train,val}.json   (from s04)
  work/<site>/split.json, labels_clean.json, frames_ok.csv                      (from s00-s03)

  1. build(): joins the three sites' train and val images into work/pooled/
     and sets the anchor sizes from the boxes of all three sites.
  2. train(): trains the tiled and/or whole-frame model on the pooled images (s05).
  3. evaluate(): scores the pooled model on each site's own test frames, with the
     threshold chosen on that site's val frames (s06, s07, s08), in work/pooled_eval/<site>/.
     Then writes work/pooled_eval/summary_all_sites.md (pooled vs per-site models).

Notebook (after the first cell has set BASE, batch_size=1, down_factor=4):
    import s12_pooled
    s12_pooled.run(cfg, BASE)                      # both models
    s12_pooled.run(cfg, BASE, modes=("down",))     # whole frame only (about 40 min)
"""
import csv, json, os, shutil, time
from pathlib import Path

import numpy as np

import s05_train, s06_infer, s07_evaluate, s08_errors
from common import coco_to_boxes, load_json, save_json
from config import paths

SITES = {"Station_1st_night": "Station_1st_night", "Station_2nd_night": "Station_2nd_night", "OAK": "OAK"}
# pooled training: three times the images, so more steps per epoch for the tiled model
POOLED = dict(epochs_tiled=15, iters_per_epoch_tiled=600, epochs_down=30)


def _anchor_sizes(base):
    sides = []
    for site in SITES:
        boxes, _ = coco_to_boxes(load_json(Path(base, "work", site, "labels_clean.json")))
        for b in boxes.values():
            if len(b):
                sides.append(np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    s = np.concatenate(sides)
    lo, hi = max(16, np.percentile(s, 2)), np.percentile(s, 99.5)     # same rule as s02_validate
    return [int(round(v / 16.0) * 16) for v in np.geomspace(lo, hi, 5)]


def build(cfg, base):
    pw = Path(base, "work", "pooled")
    cfg = dict(cfg, work_dir=str(pw), anchor_sizes=_anchor_sizes(base))
    p = paths(cfg)
    for kind in ("tiles", "downscaled"):
        for part in ("train", "val"):
            items = []
            for site in SITES:
                f = Path(base, "work", site, kind, "%s.json" % part)
                if not f.exists():
                    raise SystemExit("missing %s - run the notebook for %s up to step 4 first" % (f, site))
                items += load_json(f)
            (pw / kind).mkdir(parents=True, exist_ok=True)
            save_json(items, pw / kind / ("%s.json" % part))
            print("  pooled %-10s %-5s %5d images" % (kind, part, len(items)))
    save_json({"anchor_sizes": cfg["anchor_sizes"]}, pw / "pooled_config.json")
    print("  anchor sizes from all three sites:", cfg["anchor_sizes"])
    return cfg, p


def train(cfg, modes=("tiled", "down")):
    for m in modes:
        c = dict(cfg)
        if m == "tiled":
            c.update(epochs=POOLED["epochs_tiled"], iters_per_epoch=POOLED["iters_per_epoch_tiled"])
        else:
            c.update(epochs=POOLED["epochs_down"])
        t0 = time.time()
        s05_train.main(c, mode=m)
        print("  [%s] trained in %.0f min" % (m, (time.time() - t0) / 60))


def evaluate(cfg, base, modes=("tiled", "down")):
    pooled_models = Path(base, "work", "pooled", "models")
    rows = {}
    for site, folder in SITES.items():
        sw = Path(base, "work", site)
        ew = Path(base, "work", "pooled_eval", site)
        c = dict(cfg, work_dir=str(ew), raw_dir=os.path.join(base, "cvat_upload", folder))
        p = paths(c)
        for f in ("split.json", "labels_clean.json", "frames_ok.csv"):
            shutil.copy(sw / f, ew / f)
        for m in modes:
            shutil.copy(pooled_models / ("%s_best.pt" % m), p["models"] / ("%s_best.pt" % m))
            for part in ("val", "test"):
                s06_infer.main(c, mode=m, frames=part)
        s07_evaluate.main(c)
        for m in modes:
            s08_errors.main(c, mode=m)
        rows[site] = {"pooled": load_json(p["results"] / "summary.json"),
                      "own": load_json(sw / "results" / "summary.json") if (sw / "results" / "summary.json").exists() else {}}
        print("  %s done -> %s" % (site, p["results"]))
    _table(rows, base, modes)


def _table(rows, base, modes):
    lab = {"tiled": "tiled", "down": "whole frame"}
    keys = [("AP50", "AP50", "%.3f"), ("F1", "f1", "%.3f"), ("Recall", "recall", "%.3f"), ("Precision", "precision", "%.3f"),
            ("Count MAE / frame", "count_mae", "%.2f"), ("Count bias / frame", "count_bias", "%+.2f")]
    md = ["# Pooled model (trained on all three sites) vs per-site models", "",
          "Each site is scored on its own 24 test frames; thresholds chosen on that site's val frames.", ""]
    for m in modes:
        md += ["## %s" % lab[m], "", "| Test metric | " + " | ".join("%s per-site | %s pooled" % (s, s) for s in rows) + " |",
               "|---|" + "---|---|" * len(rows)]
        for name, k, fmt in keys:
            cells = []
            for s, r in rows.items():
                own = r["own"].get(m, {}).get(k); pool = r["pooled"].get(m, {}).get(k)
                cells += [fmt % own if own is not None else "-", fmt % pool if pool is not None else "-"]
            md.append("| %s | %s |" % (name, " | ".join(cells)))
        rec = []
        for s, r in rows.items():
            for src in ("own", "pooled"):
                v = r[src].get(m, {}).get("recall_by_size", {}).get("small <100px", {}).get("recall")
                rec.append("%.2f" % v if v is not None else "-")
        md += ["| Recall small <100 px | %s |" % " | ".join(rec), ""]
    out = Path(base, "work", "pooled_eval", "summary_all_sites.md")
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md)); print("->", out)


def run(cfg, base, modes=("tiled", "down")):
    cfg_p, _ = build(cfg, base)
    train(cfg_p, modes)
    evaluate(cfg_p, base, modes)
