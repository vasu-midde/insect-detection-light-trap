"""Step 0 - check every frame in raw_dir and write frames_ok.csv.
The Oak copy on D: has broken frames (0200 onward), so nothing downstream should ever touch an
unchecked file. Run once; takes ~1 min for 510 frames (reads only the first and last bytes)."""
import csv
import sys

from common import check_jpeg, frame_index, frame_time, list_frames
from config import CFG, paths


def main(cfg=CFG):
    p = paths(cfg)
    frames = list_frames(cfg["raw_dir"])
    if not frames:
        sys.exit("No frames found under %s" % cfg["raw_dir"])
    rows, bad = [], 0
    for f in frames:
        ok, why = check_jpeg(f)
        bad += not ok
        rows.append({"index": frame_index(f.name), "time": frame_time(f.name), "file": str(f),
                     "ok": int(ok), "reason": why})
    with open(p["frames_csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    good = [r for r in rows if r["ok"]]
    print("frames: %d  good: %d  broken: %d" % (len(rows), len(good), bad))
    if good:
        print("good range: #%04d (%s) .. #%04d (%s)" % (good[0]["index"], good[0]["time"],
                                                      good[-1]["index"], good[-1]["time"]))
    for r in rows:
        if not r["ok"]:
            print("  broken #%04d  %s" % (r["index"], r["reason"]))
            if bad > 20:
                print("  ... (see %s)" % p["frames_csv"]); break
    return rows


if __name__ == "__main__":
    main()
