"""W1 task 1: where do SuperAnimal-Bird's leg points land on Duong's annotated frames?

Only official DeepLabCut functions are used for inference:
  1. deeplabcut.utils.pseudo_label.keypoint_matching  -> official conversion table
     (Duong 16 points <-> superanimal_bird 42 points) + confusion matrix
  2. deeplabcut.pose_estimation_pytorch.apis.analyze_images.superanimal_analyze_images
     -> raw SA predictions on every human-labeled frame
This script only adds the bookkeeping: a working copy of the Duong project (the original
data is never modified) and a distance table between SA leg points and Duong leg points.

Outputs (all under --out):
  project/                         working copy (config.yaml + symlinks)
  project/memory_replay/conversion_table.csv, confusion_matrix.png   (official)
  sa_labeled_images/               SA predictions drawn on frames (official)
  sa_predictions.json              raw SA predictions
  leg_distances.csv                one row per (frame, SA leg point, Duong leg point)
  summary.md                       median distances and nearest-joint shares
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SA = "superanimal_bird"
SA_POSE = "resnet_50"
SA_DET = "fasterrcnn_mobilenet_v3_large_fpn"
SA_LEG = ["left_leg", "right_leg", "left_tibia", "right_tibia", "left_foot", "right_foot"]
DUONG_LEG = [f"{s}_{j}" for s in ("left", "right") for j in ("hip", "knee", "ankle", "foot", "foot_tip")]
# anatomical names used in TARSUS (Duong names are kept in the data)
ANAT = {"hip": "hip", "knee": "knee", "ankle": "itj", "foot": "tmp", "foot_tip": "toe"}


def make_working_project(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    lines = (src / "config.yaml").read_text(encoding="utf-8").splitlines()
    lines = [f"project_path: {dst}" if l.startswith("project_path:") else l for l in lines]
    (dst / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for sub in ("labeled-data", "videos"):
        link = dst / sub
        if not link.exists():
            link.symlink_to(src / sub, target_is_directory=True)
    return dst / "config.yaml"


def load_labels(proj: Path) -> pd.DataFrame:
    """All human labels, one row per image, columns (bodypart, coord)."""
    frames = []
    for csv in sorted((proj / "labeled-data").glob("*/CollectedData_*.csv")):
        df = pd.read_csv(csv, header=[0, 1, 2], index_col=[0, 1, 2])
        df.columns = df.columns.droplevel(0)  # drop scorer
        df.index = [str(proj / "labeled-data" / a / b) for _, a, b in df.index]
        frames.append(df)
    return pd.concat(frames)


def md(df) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duong-project", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg_path = make_working_project(args.duong_project, out / "project")

    # 1. official keypoint matching
    from deeplabcut.utils.pseudo_label import keypoint_matching

    mr = out / "project" / "memory_replay"
    if (mr / "conversion_table.csv").exists():
        print("reuse existing official conversion table")
    else:
        if mr.exists():
            shutil.rmtree(mr)
        keypoint_matching(str(cfg_path), SA, SA_POSE, SA_DET, device=args.device)
    print("official conversion table:", mr / "conversion_table.csv")

    # 2. official SA inference on every labeled frame
    from deeplabcut.pose_estimation_pytorch.apis.analyze_images import superanimal_analyze_images
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config

    labels = load_labels(out / "project")
    labels = labels[~labels.index.duplicated()]
    images = [p for p in labels.index if Path(p).exists()]
    print(f"labeled frames: {len(labels)}, found on disk: {len(images)}")
    sa_names = load_super_animal_config(SA, SA_POSE, SA_DET)["metadata"]["bodyparts"]
    pred_file = out / "sa_predictions.json"
    if pred_file.exists():
        print("reuse existing SA predictions")
        preds = json.loads(pred_file.read_text())
    else:
        preds = superanimal_analyze_images(
            SA, SA_POSE, SA_DET, images, max_individuals=1,
            out_folder=out / "sa_labeled_images", device=args.device,
            pose_threshold=0.1, bbox_threshold=0.1,
        )
        with pred_file.open("w") as f:
            json.dump({k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in preds.items()}, f)

    # 3. distances between SA leg points and Duong leg points
    rows = []
    for img in images:
        p = preds.get(img)
        if p is None or len(p["bodyparts"]) == 0:
            continue
        kp = np.asarray(p["bodyparts"])[0]  # (42, 3) x, y, score
        lab = labels.loc[img]
        xy = lab.unstack()[["x", "y"]].astype(float)
        valid = xy.dropna()
        if len(valid) < 2:
            continue
        diag = float(np.hypot(*(valid.max() - valid.min()).values))
        if diag < 1.0:  # degenerate label set (all points on one spot)
            continue
        for sp in SA_LEG:
            sx, sy, sc = kp[sa_names.index(sp)]
            for dp in DUONG_LEG:
                if dp not in valid.index:
                    continue
                d = float(np.hypot(sx - valid.loc[dp, "x"], sy - valid.loc[dp, "y"]))
                rows.append(dict(image=img, species=Path(img).parent.name.split("_")[0], sa_point=sp,
                                 sa_score=float(sc), duong_point=dp, dist_px=d, dist_norm=d / diag))
    dist = pd.DataFrame(rows)
    dist.to_csv(out / "leg_distances.csv", index=False)

    # 4. summary
    med = dist.pivot_table(index="sa_point", columns="duong_point", values="dist_norm", aggfunc="median")
    med = med.reindex(index=SA_LEG, columns=[c for c in DUONG_LEG if c in med.columns])
    nearest = dist.loc[dist.groupby(["image", "sa_point"])["dist_norm"].idxmin()]
    nearest = nearest.assign(joint=nearest.duong_point.str.replace("left_", "").str.replace("right_", "").map(ANAT))
    share = pd.crosstab(nearest.sa_point, nearest.joint, normalize="index").reindex(SA_LEG)
    score = dist.drop_duplicates(["image", "sa_point"]).groupby("sa_point").sa_score.median().reindex(SA_LEG)
    with (out / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# W1 superanimal_bird leg points vs Duong labels\n\n")
        f.write(f"frames analysed: {dist.image.nunique()}\n\n")
        f.write("## Share of frames in which each SA point is nearest to each joint (side ignored)\n\n")
        f.write(md(share.round(3)) + "\n\n")
        f.write("## Median distance / GT-bbox diagonal\n\n" + md(med.round(3)) + "\n\n")
        f.write("## Median SA confidence\n\n" + md(score.round(3)) + "\n")
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
