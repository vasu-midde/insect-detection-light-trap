"""Step 4 - make the training images for both models.

TILED model : frame shrunk by tile_scale (1 = native), cut into tile x tile tiles with overlap px. Every tile with an insect is kept,
              plus empty_tile_ratio empty tiles per positive tile (sheet texture, stains, info bar
              = the false alarms the model must learn to ignore).
              A box cut by the tile edge is kept (clipped) if >= 50 % of it is inside the tile.
DOWNSCALED  : the whole frame shrunk by down_factor (baseline: is tiling worth it?).

Only train and val frames are written - test frames are only ever seen by s06_infer, whole.
"""
import random

import cv2
import numpy as np

from common import area, coco_to_boxes, list_frames, load_json, read_rgb, save_json, tile_grid
from config import CFG, paths


def _write(img_rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main(cfg=CFG, splits=("train", "val")):
    p = paths(cfg)
    rng = random.Random(cfg["seed"])
    boxes, _ = coco_to_boxes(load_json(p["labels"]))
    split = load_json(p["split"])
    raw = {f.name: f for f in list_frames(cfg["raw_dir"])}
    T, O, D, S = cfg["tile"], cfg["overlap"], cfg["down_factor"], cfg["tile_scale"]

    for s in splits:
        tiles_ann, down_ann = [], []
        n_pos = n_neg = 0
        for k, name in enumerate(split[s], 1):
            b_native = boxes[name]
            # ---- downscaled whole frame (libjpeg DCT scaling - the same read is used at inference)
            small = read_rgb(raw[name], reduce=D)
            out = p["down"] / s / name
            _write(small, out)
            down_ann.append({"file": str(out), "boxes": (b_native / D).round(1).tolist()})
            img = small if S == D else read_rgb(raw[name], reduce=S)
            H, W = img.shape[:2]
            b = b_native / S
            # ---- tiles
            pos, neg = [], []
            for (x, y) in tile_grid(W, H, T, O):
                if len(b):
                    c = b.copy()
                    c[:, [0, 2]] = (c[:, [0, 2]] - x).clip(0, T)
                    c[:, [1, 3]] = (c[:, [1, 3]] - y).clip(0, T)
                    vis = area(c) / np.maximum(area(b), 1)
                    c = c[vis >= 0.5]
                else:
                    c = np.zeros((0, 4), np.float32)
                (pos if len(c) else neg).append((x, y, c))
            keep_neg = rng.sample(neg, min(len(neg), max(2, int(round(len(pos) * cfg["empty_tile_ratio"])))))
            for (x, y, c) in pos + keep_neg:
                out = p["tiles"] / s / ("%s_x%05d_y%05d.jpg" % (name[:-4], x, y))
                _write(img[y:y + T, x:x + T], out)
                tiles_ann.append({"file": str(out), "frame": name, "x": x, "y": y, "boxes": c.round(1).tolist()})
            n_pos += len(pos); n_neg += len(keep_neg)
            if k % 10 == 0 or k == len(split[s]):
                print("  %s: %d/%d frames, %d positive + %d empty tiles" % (s, k, len(split[s]), n_pos, n_neg))
        save_json(tiles_ann, p["tiles"] / ("%s.json" % s))
        save_json(down_ann, p["down"] / ("%s.json" % s))
    print("tiles      ->", p["tiles"])
    print("downscaled ->", p["down"])


if __name__ == "__main__":
    main()
