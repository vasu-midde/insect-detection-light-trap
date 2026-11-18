"""Step 3 - train / val / test split in contiguous time blocks.

Frames one minute apart are near-copies (a resting moth does not move), so a random split would put
almost the same picture in train and test. Instead the night is cut into n_blocks contiguous blocks
and each block gets one role (block_roles). Val and test blocks sit in different parts of the night.
gap_frames drops frames at a block edge where the role changes, so neighbours never straddle a split.
"""
from collections import Counter

from common import coco_to_boxes, frame_index, frame_time, load_json, save_json
from config import CFG, paths


def main(cfg=CFG):
    p = paths(cfg)
    boxes, _ = coco_to_boxes(load_json(p["labels"]))
    names = sorted(boxes, key=frame_index)
    n, k = len(names), cfg["n_blocks"]
    roles = cfg["block_roles"]
    assert len(roles) == k, "block_roles must have n_blocks entries"
    block_of = [min(i * k // n, k - 1) for i in range(n)]
    role = [roles[b] for b in block_of]

    drop = set()
    g = cfg["gap_frames"]
    for i in range(1, n):
        if role[i] != role[i - 1]:
            for j in range(max(0, i - g), min(n, i + g)):
                # drop from the training side so val/test keep every frame they can
                if role[j] == "train":
                    drop.add(j)

    split = {"train": [], "val": [], "test": [], "dropped_gap": []}
    for i, name in enumerate(names):
        split["dropped_gap" if i in drop else role[i]].append(name)
    save_json(split, p["split"])

    for s in ("train", "val", "test"):
        nb = sum(len(boxes[x]) for x in split[s])
        t = [frame_time(x)[11:16] for x in split[s]]
        blocks = sorted(set(block_of[names.index(x)] for x in split[s]))
        print("%-5s %3d frames  %4d insects  blocks %s  (%s ... %s)" % (s, len(split[s]), nb, blocks, t[0] if t else "-", t[-1] if t else "-"))
    print("gap   %3d frames dropped at split edges" % len(split["dropped_gap"]))
    if not any(len(boxes[x]) for x in split["test"]):
        print("WARNING: test split has no insects - change block_roles so a later block is test")
    return split


if __name__ == "__main__":
    main()
