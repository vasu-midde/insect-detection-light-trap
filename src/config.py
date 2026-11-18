"""
One place for every setting. Edit the PATHS block on the lab machine; leave the rest
unless you have a reason. All scripts read CFG from here; the notebook can override it.
"""
from pathlib import Path

CFG = {
    # ---------------- PATHS (edit these) ----------------
    # Folder that holds the Oak frames (searched recursively for *.JPG)
    "raw_dir": r"D:\Dataset\Oak",
    # Where everything this pipeline makes goes (tiles, models, results)
    "work_dir": r"D:\Dataset\work_oak",
    # CVAT export (COCO 1.0) after you corrected the pre-labels
    "cvat_coco": r"D:\Dataset\work_oak\cvat_export\annotations\instances_default.json",

    # ---------------- frames ----------------
    "frame_size": (5120, 2880),     # CHANGED: frames are the half-resolution CVAT copies (full camera frame is 10240 x 5760)
    "n_pilot": 120,                 # frames to annotate, spread evenly over the good part of the night
    "info_bar_frac": 0.08,          # bottom 8 % of the frame is the camera's info bar -> ignored
    "label_box_frac": (0.94, 0.86, 1.0, 1.0),  # paper label at bottom-right (x0,y0,x1,y1 as fractions) -> ignored by pre-labeller

    # ---------------- split (time blocks) ----------------
    "n_blocks": 10,                 # night cut into 10 contiguous blocks
    "block_roles": ["train", "train", "val", "train", "test",
                    "train", "train", "val", "train", "test"],
    "gap_frames": 1,                # drop 1 frame at each block edge where the split changes (less leakage)

    # ---------------- tiling ----------------
    # Frames are shrunk by tile_scale, then cut into overlapping tiles. Midges (~50 px) stay ~25 px,
    # the biggest moths (~1300 px) become ~650 px. tile_scale = 1 is full native resolution
    # (better for the tiniest insects but ~4x slower to train and run).
    # overlap (in shrunk pixels) must be >= the largest insect so every insect is whole in one tile.
    "tile_scale": 1,                # CHANGED: the half-res frames are already shrunk 2x, so tiles are cut without further shrinking
    "tile": 1280,
    "overlap": 704,
    "empty_tile_ratio": 0.3,        # keep this many empty tiles per positive tile for training (hard negatives)

    # ---------------- downscaled baseline ----------------
    "down_factor": 2,               # CHANGED: half-res frame shrunk 2x -> 2560 x 1440 (same as before)

    # ---------------- model / training ----------------
    "epochs": 30,                   # CHANGED (was 12): tiled model was still improving at epoch 12
    "iters_per_epoch": 200,         # random sample of the training images each epoch (frames are near-duplicates)
    "max_minutes": None,            # CHANGED (was {"tiled": 40, "down": 20}): no time cap, all epochs run; set a number to cap again
    "val_max_images": 250,          # val tiles/frames scored after each epoch (model selection)
    "batch_size": 2,
    "lr": 0.01,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "amp": True,
    # Anchor sizes per FPN level (P2..P6) in NATIVE frame pixels; divided by tile_scale / down_factor
    # inside the model. s02_validate prints a suggestion from your real boxes.
    "anchor_sizes": [32, 64, 128, 256, 512],   # CHANGED: halved for half-res frames (s02 replaces them with measured sizes)
    "anchor_ratios": [0.5, 1.0, 2.0],
    "seed": 0,

    # ---------------- inference / merge ----------------
    "score_floor": 0.05,            # keep detections above this before merging (threshold is chosen later on val)
    "merge_iou": 0.5,
    "merge_containment": 0.7,       # a box that lies >=70 % inside a higher-scoring box is a duplicate
    "tile_batch": 8,                # halves itself automatically on GPU out-of-memory
    "eval_iou": 0.5,                # IoU for counting a detection as correct
}


def paths(cfg=CFG):
    w = Path(cfg["work_dir"])
    p = {
        "work": w,
        "frames_csv": w / "frames_ok.csv",
        "pilot_dir": w / "pilot_frames",
        "prelabels": w / "prelabels_coco.json",
        "labels": w / "labels_clean.json",
        "stats": w / "label_stats.json",
        "split": w / "split.json",
        "tiles": w / "tiles",
        "down": w / "downscaled",
        "models": w / "models",
        "preds": w / "predictions",
        "results": w / "results",
    }
    for k in ("work", "models", "preds", "results"):
        p[k].mkdir(parents=True, exist_ok=True)
    return p
