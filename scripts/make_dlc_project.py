"""Build the TARSUS working DLC project from Duong's project (original data is never modified).

- outputs/dlc_project/config.yaml : copy of Duong's config with project_path set, iteration 0,
  and video_sets covering ALL labeled-data folders that have a CollectedData csv (Duong's
  config lists only 22 of the 35, the other 13 would be silently dropped by DLC).
- outputs/dlc_project/labeled-data, videos : symlinks into the Duong project.
- outputs/dlc_project/split.json : species leave-one-out folds (train/test row indices into
  the merged label table, built with DLC's own merge_annotateddatasets so the order matches).
- outputs/dlc_project/index_map.csv : merged row -> folder, video id, species.
Usage: python scripts/make_dlc_project.py --duong-project <...> --out outputs/dlc_project
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

SPECIES = ["Duck", "Goose", "Pelican", "Swan"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duong-project", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    src, dst = a.duong_project.resolve(), a.out.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("labeled-data", "videos"):
        if not (dst / sub).exists():
            (dst / sub).symlink_to(src / sub, target_is_directory=True)

    cfg = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
    cfg["project_path"] = str(dst)
    cfg["iteration"] = 0
    cfg["engine"] = "pytorch"
    video_sets = {}
    folders = sorted(p.parent.name for p in (src / "labeled-data").glob("*/CollectedData_*.csv"))
    for f in folders:
        png = next((src / "labeled-data" / f).glob("*.png"))
        w, h = Image.open(png).size
        video_sets[str(dst / "videos" / f"{f}.mp4")] = {"crop": f"0, {w}, 0, {h}"}
    cfg["video_sets"] = video_sets
    cfg["TrainingFraction"] = [0.9]  # extended per fold by train_eval.py
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"config written: {len(folders)} labeled folders")

    # merged label table in DLC's own order
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
    from deeplabcut.utils import auxiliaryfunctions as af

    pcfg = af.read_config(str(dst / "config.yaml"))
    tsf = dst / af.get_training_set_folder(pcfg)
    tsf.mkdir(parents=True, exist_ok=True)
    data = merge_annotateddatasets(pcfg, tsf)
    rows = []
    for i, idx in enumerate(data.index):
        folder = idx[1] if isinstance(idx, tuple) else Path(str(idx)).parts[-2]
        img = idx[2] if isinstance(idx, tuple) else Path(str(idx)).name
        vid = re.match(r"([A-Za-z]+_\d+[ab]?)", folder).group(1)
        rows.append({"row": i, "folder": folder, "image": img, "video_id": vid, "species": folder.split("_")[0]})
    im = pd.DataFrame(rows)
    im.to_csv(dst / "index_map.csv", index=False)
    split = {}
    for sp in SPECIES:
        te = im.row[im.species == sp].tolist()
        tr = im.row[im.species != sp].tolist()
        split[sp] = {"test_species": sp, "train_idx": tr, "test_idx": te,
                     "fraction": round(len(tr) / (len(tr) + len(te)), 2),
                     "test_folders": sorted(im.folder[im.species == sp].unique().tolist())}
        print(f"fold {sp:8s} train {len(tr):4d} test {len(te):4d} fraction {split[sp]['fraction']}")
    (dst / "split.json").write_text(json.dumps(split, indent=1))
    print(f"merged rows: {len(im)}; species counts: {im.species.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
