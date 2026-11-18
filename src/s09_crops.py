"""s09_crops.py - cut insect crops for the species classifier.

Two steps:

  1) python s09_crops.py cut [--site OAK]
     Cuts every annotated insect larger than MIN_SIDE px (half-res coordinates)
     out of the ORIGINAL full-resolution frames.
       species_crops/crops/<site>/...      all crops (one per box per frame)
       species_crops/to_sort/<site>/...    ONE crop per insect track (the largest)
       species_crops/sorted/<species>/     13 empty folders to sort into
       species_crops/crops_index.csv       one row per crop

  2) Move each image from to_sort/<site>/ into the right sorted/<species>/ folder.
     (Delete or leave the ones you cannot decide; leftovers are reported.)

  3) python s09_crops.py collect
     Gives every crop the species of its track's sorted image and writes
       species_crops/labels.csv            crop path, site, frame, track, species
     and prints the number of crops and tracks per species.
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # D:\datasett
OUT = os.path.join(ROOT, "species_crops")
SITES = {   # site -> (folder with original frames, annotation file in half-res coordinates)
    "Station_1st_night": ("Station_1st_night", "Station_1st_night_coco.json"),
    "Station_2nd_night": ("Station_2nd_night", "Station_2nd_night_coco.json"),
    "OAK":               ("OAK",               "instances_default[OAK].json"),
}
SPECIES = ["Deltote deceptoria", "Plagodis", "Chiasmia", "Thaumetopoea processionea",
           "Lomaspilis marginata", "Calliteara pudibunda",
           "Eilema sp.", "Heliothis sp.", "Drepaninae",                     # identified by Prof. Birkhofer (Nov 2026)
           "Morphospecies 1", "Morphospecies 2", "Morphospecies 3",       # were species_0109, species_0132, species_N2_0113
           "other insect"]
MIN_SIDE = 100     # keep boxes whose longer side is > 100 px in half-res coordinates
SCALE = 2          # half-res annotation -> full-res original
PAD = 0.15         # margin around the box, as a share of the box size
MAX_OUT = 768      # longest side of a saved crop (smaller crops keep their size)
INDEX = os.path.join(OUT, "crops_index.csv")
FIELDS = ["crop", "site", "frame", "track", "ann_id", "x", "y", "w", "h", "is_rep"]


def frame_no(name):
    m = re.search(r"_(\d{4})\.jpe?g$", name, re.I)
    return int(m.group(1)) if m else -1


def find_original(folder, name):
    """Annotation names end in .jpg, originals in .JPG: match without case."""
    lookup = {f.lower(): f for f in os.listdir(folder)}
    return lookup.get(name.lower())


def cut_site(site):
    img_dir, ann_file = SITES[site]
    img_dir = os.path.join(ROOT, img_dir)
    coco = json.load(open(os.path.join(ROOT, "annotations", ann_file)))
    images = {im["id"]: im for im in coco["images"]}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        if max(a["bbox"][2], a["bbox"][3]) > MIN_SIDE:
            by_img[a["image_id"]].append(a)

    crop_dir = os.path.join(OUT, "crops", site)
    os.makedirs(crop_dir, exist_ok=True)
    rows = []
    for k, (img_id, anns) in enumerate(sorted(by_img.items())):
        im_info = images[img_id]
        fname = find_original(img_dir, im_info["file_name"])
        if fname is None:
            print("  missing original:", im_info["file_name"]); continue
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        W, H = img.size
        f = frame_no(fname)
        for a in anns:
            x, y, w, h = [v * SCALE for v in a["bbox"]]
            px, py = PAD * w, PAD * h
            box = (max(0, int(x - px)), max(0, int(y - py)),
                   min(W, int(x + w + px)), min(H, int(y + h + py)))
            crop = img.crop(box)
            crop.thumbnail((MAX_OUT, MAX_OUT))
            tid = a.get("attributes", {}).get("track_id")
            track = f"t{tid:04d}" if tid is not None else f"s{a['id']:05d}"   # s = single box, no track
            name = f"{site}_{track}_f{f:04d}_a{a['id']:05d}.jpg"
            crop.save(os.path.join(crop_dir, name), quality=92)
            rows.append(dict(crop=f"crops/{site}/{name}", site=site, frame=f, track=track,
                             ann_id=a["id"], x=round(x), y=round(y), w=round(w), h=round(h), is_rep=0))
        print(f"  {site}: frame {k + 1}/{len(by_img)}", end="\r")
    print()

    # one representative per track: the largest box
    tracks = defaultdict(list)
    for r in rows:
        tracks[r["track"]].append(r)
    rep_dir = os.path.join(OUT, "to_sort", site)
    os.makedirs(rep_dir, exist_ok=True)
    for track, rs in tracks.items():
        best = max(rs, key=lambda r: r["w"] * r["h"])
        best["is_rep"] = 1
        src = os.path.join(OUT, best["crop"])
        dst = os.path.join(rep_dir, f"{site}_{track}_n{len(rs)}.jpg")
        Image.open(src).save(dst, quality=95)
    print(f"  {site}: {len(rows)} crops, {len(tracks)} tracks to sort")
    return rows


def cmd_cut(sites):
    for s in SPECIES:
        os.makedirs(os.path.join(OUT, "sorted", s), exist_ok=True)
    old = []
    if os.path.exists(INDEX):   # keep rows of sites not re-cut now
        old = [r for r in csv.DictReader(open(INDEX)) if r["site"] not in sites]
    rows = old
    for site in sites:
        rows += cut_site(site)
    with open(INDEX, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    print("index:", INDEX)


def cmd_collect():
    rows = list(csv.DictReader(open(INDEX)))
    label = {}   # (site, track) -> species
    for sp in SPECIES:
        for f in os.listdir(os.path.join(OUT, "sorted", sp)):
            m = re.match(r"(.+)_([ts]\d+)_n\d+\.jpg$", f)
            if m:
                label[(m.group(1), m.group(2))] = sp
    out = os.path.join(OUT, "labels.csv")
    n_crops, n_tracks, unsorted = defaultdict(int), defaultdict(set), set()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["crop", "site", "frame", "track", "species"])
        for r in rows:
            sp = label.get((r["site"], r["track"]))
            if sp is None:
                unsorted.add((r["site"], r["track"])); continue
            w.writerow([r["crop"], r["site"], r["frame"], r["track"], sp])
            n_crops[sp] += 1; n_tracks[sp].add((r["site"], r["track"]))
    print(f"{'species':28s} {'tracks':>6s} {'crops':>6s}")
    for sp in SPECIES:
        print(f"{sp:28s} {len(n_tracks[sp]):6d} {n_crops[sp]:6d}")
    print(f"not sorted yet: {len(unsorted)} tracks")
    print("labels:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["cut", "collect"])
    p.add_argument("--site", choices=list(SITES), help="cut one site only (default: all)")
    a = p.parse_args()
    if a.step == "cut":
        cmd_cut([a.site] if a.site else list(SITES))
    else:
        cmd_collect()
