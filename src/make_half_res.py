"""Make the half-resolution copies of the annotated frames from the full-size originals.
They are the same frames you annotated in CVAT (5120 x 2880, made the same way), so the
CVAT boxes fit them exactly. Files that already exist are skipped, so it is safe to run again."""
import os
from pathlib import Path

import cv2

from common import load_json


def main(orig_dir, coco_path, out_dir, quality=85):
    wanted = [Path(im["file_name"]).name for im in load_json(coco_path)["images"]]
    found = {p.name.lower(): p for p in Path(orig_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")}
    missing = [n for n in wanted if n.lower() not in found]
    if missing:
        raise SystemExit("%d annotated frames not found under %s, e.g. %s" % (len(missing), orig_dir, missing[:3]))
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for k, n in enumerate(wanted, 1):
        src = found[n.lower()]
        dst = Path(out_dir) / src.name
        if not dst.exists():
            im = cv2.imread(str(src), cv2.IMREAD_REDUCED_COLOR_2)
            if im is None or im.shape[:2] != (2880, 5120):
                raise SystemExit("cannot read %s as a 10240 x 5760 frame" % src)
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
            made += 1
        if k % 20 == 0 or k == len(wanted):
            print("  %d/%d frames ready" % (k, len(wanted)))
    print("half-res frames ->", out_dir, "(%d new)" % made)
