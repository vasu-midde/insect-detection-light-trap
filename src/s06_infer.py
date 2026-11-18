"""Step 6 - run a trained model on whole frames.

tiled: frame shrunk by tile_scale -> overlapping tiles -> detect -> shift + scale boxes back -> IoU + containment merge
down : frame shrunk by down_factor -> detect once -> boxes scaled back up
Writes predictions/<mode>_<setname>.json : {frame: {"boxes": [...], "scores": [...], "seconds": t}}
All boxes above score_floor are kept; the counting threshold is chosen later, on val only.
"""
import time

import cv2
import numpy as np
import torch

from common import frame_index, list_frames, load_json, merge_boxes, read_rgb, save_json, tile_grid
from config import CFG, paths
from model import load_model


def _to_tensor(img, device):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).to(device)


@torch.no_grad()
def detect_frame(model, img, cfg, mode, device, state):
    """img is the frame already shrunk (down_factor for 'down', tile_scale for 'tiled').
    Returns boxes in NATIVE frame pixels."""
    H, W = img.shape[:2]
    if mode == "down":
        o = model([_to_tensor(img, device)])[0]
        return o["boxes"].cpu().numpy() * cfg["down_factor"], o["scores"].cpu().numpy()

    T = cfg["tile"]
    corners = tile_grid(W, H, T, cfg["overlap"])
    allb, alls = [], []
    k = 0
    while k < len(corners):
        chunk = corners[k:k + state["tile_batch"]]
        try:
            out = model([_to_tensor(img[y:y + T, x:x + T], device) for (x, y) in chunk])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if state["tile_batch"] == 1:
                raise
            state["tile_batch"] = max(1, state["tile_batch"] // 2)      # remembered for later frames
            print("  GPU memory: tile batch -> %d" % state["tile_batch"])
            continue
        for (x, y), o in zip(chunk, out):
            s = o["scores"].cpu().numpy()
            m = s >= cfg["score_floor"]
            b = o["boxes"].cpu().numpy()[m]
            # CHANGED: drop boxes cut by an inner tile edge - the overlap is larger than any insect,
            # so the whole insect is also seen, uncut, in a neighbouring tile
            e = 3
            cut = (((b[:, 0] < e) & (x > 0)) | ((b[:, 1] < e) & (y > 0)) |
                   ((b[:, 2] > T - e) & (x + T < W)) | ((b[:, 3] > T - e) & (y + T < H)))
            b, s = b[~cut], s[m][~cut]
            b[:, [0, 2]] += x; b[:, [1, 3]] += y
            allb.append(b); alls.append(s)
        k += len(chunk)
    b = np.concatenate(allb) if allb else np.zeros((0, 4), np.float32)
    s = np.concatenate(alls) if alls else np.zeros(0, np.float32)
    keep = merge_boxes(b, s, cfg["merge_iou"], cfg["merge_containment"])
    return b[keep] * cfg["tile_scale"], s[keep]


def main(cfg=CFG, mode="tiled", frames="test", tag=None):
    """frames: 'val' / 'test' (from split.json), 'all' (every good frame), or a list of file names."""
    p = paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(p["models"] / ("%s_best.pt" % mode), cfg, mode, device)
    raw = {f.name: f for f in list_frames(cfg["raw_dir"])}
    if frames in ("val", "test"):
        names = load_json(p["split"])[frames]
    elif frames == "all":
        import csv
        names = [r["file"].replace("\\", "/").split("/")[-1] for r in csv.DictReader(open(p["frames_csv"], encoding="utf-8")) if r["ok"] == "1"]
    else:
        names = list(frames)
    names = sorted(names, key=frame_index)
    tag = tag or (frames if isinstance(frames, str) else "custom")

    state = {"tile_batch": cfg["tile_batch"]}
    out, t_all = {}, time.time()
    for k, name in enumerate(names, 1):
        img = read_rgb(raw[name], reduce=cfg["down_factor"] if mode == "down" else cfg["tile_scale"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        b, s = detect_frame(model, img, cfg, mode, device, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[name] = {"boxes": np.round(b, 1).tolist(), "scores": np.round(s, 4).tolist(), "seconds": round(time.time() - t, 3)}
        if k % 10 == 0 or k == len(names):
            print("  [%s/%s] %d/%d frames  %.1f s/frame" % (mode, tag, k, len(names), np.mean([v["seconds"] for v in out.values()])))
    path = p["preds"] / ("%s_%s.json" % (mode, tag))
    save_json(out, path)
    print("predictions -> %s  (%.0f s total)" % (path, time.time() - t_all))
    return out


def load_preds(path):
    d = load_json(path)
    return {k: (np.array(v["boxes"], np.float32).reshape(-1, 4), np.array(v["scores"], np.float32)) for k, v in d.items()}, \
           {k: v["seconds"] for k, v in d.items()}


if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "tiled", frames=sys.argv[2] if len(sys.argv) > 2 else "test")
