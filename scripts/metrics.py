"""Metrics for TARSUS runs: PCK / RMSE on the test fold, within-video bone-ratio CV.

GT and predictions are DLC DataFrames aligned on the same index (evaluate_network reindexes
predictions to the merged label table). Normalisation:
  diag : diagonal of the bounding box of all human-labeled points in the frame (primary)
  tmt  : human itj->tmp length (mean of the sides present in the frame)     (secondary)
PCK counts every GT point; a missing prediction is a miss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = {"knee": "knee", "hip": "hip", "ankle": "itj", "foot": "tmp", "foot_tip": "toe"}
VISIBLE = ["ankle", "foot", "foot_tip"]
THRESH = (0.05, 0.1, 0.2)          # fraction of GT bbox diagonal
THRESH_TMT = (0.25, 0.5, 1.0)      # multiples of the GT tarsometatarsus length


def _xy(df: pd.DataFrame, bp: str) -> np.ndarray:
    sc = df.columns.get_level_values(0)[0]
    return df[(sc, bp, "x")].to_numpy(float), df[(sc, bp, "y")].to_numpy(float)


def compute(gt: pd.DataFrame, pred: pd.DataFrame, test_rows: list[int], index_map: pd.DataFrame) -> dict:
    gt = gt.iloc[test_rows]
    pred = pred.reindex(gt.index)
    im = index_map.set_index("row").loc[test_rows]
    sc = gt.columns.get_level_values(0)[0]
    bps = [b for b in gt.columns.get_level_values(1).unique()]
    # per-frame normalisers
    xs = np.stack([gt[(sc, b, "x")].to_numpy(float) for b in bps], 1)
    ys = np.stack([gt[(sc, b, "y")].to_numpy(float) for b in bps], 1)
    diag = np.hypot(np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1))
    tmt = []
    for side in ("left", "right"):
        ax, ay = _xy(gt, f"{side}_ankle")
        fx, fy = _xy(gt, f"{side}_foot")
        tmt.append(np.hypot(ax - fx, ay - fy))
    tmt = np.nanmean(np.stack(tmt, 1), 1)

    out = {"n_test_frames": int(len(gt)), "per_joint": {}}
    dist_all = {}
    for joint, anat in JOINTS.items():
        d, nd, nt, sp = [], [], [], []
        for side in ("left", "right"):
            gx, gy = _xy(gt, f"{side}_{joint}")
            px, py = _xy(pred, f"{side}_{joint}")
            dd = np.hypot(gx - px, gy - py)
            ok = np.isfinite(gx) & np.isfinite(gy)
            dd = np.where(np.isfinite(dd), dd, np.inf)[ok]
            d.append(dd)
            nd.append(dd / diag[ok])
            nt.append(dd / tmt[ok])
            sp.append(im.species.to_numpy()[ok])
        d, nd, nt, sp = map(np.concatenate, (d, nd, nt, sp))
        dist_all[joint] = (d, nd, sp)
        r = {"n": int(len(d)), "rmse_px": float(np.sqrt(np.mean(np.minimum(d, 1e6) ** 2))),
             "median_px": float(np.median(d)), "anat": anat}
        for t in THRESH:
            r[f"pck@{t}_diag"] = float(np.mean(nd < t))
        for t in THRESH_TMT:
            r[f"pck@{t}tmt"] = float(np.mean(nt < t))
        for s in np.unique(sp):
            r[f"pck@0.1_diag_{s}"] = float(np.mean(nd[sp == s] < 0.1))
        out["per_joint"][joint] = r
    vis = np.concatenate([dist_all[j][0] for j in VISIBLE])
    out["visible_rmse_px"] = float(np.sqrt(np.mean(np.minimum(vis, 1e6) ** 2)))
    out["visible_rmse_diag"] = float(np.sqrt(np.mean(np.minimum(np.concatenate([dist_all[j][1] for j in VISIBLE]), 1e3) ** 2)))

    # within-video CV of predicted tib/tmt (and GT, for reference)
    def cv_table(df):
        rows = []
        for side in ("left", "right"):
            kx, ky = _xy(df, f"{side}_knee"); ax, ay = _xy(df, f"{side}_ankle"); fx, fy = _xy(df, f"{side}_foot")
            tib, tm = np.hypot(kx - ax, ky - ay), np.hypot(ax - fx, ay - fy)
            r = tib / tm
            for v, val in zip(im.video_id.to_numpy(), r):
                if np.isfinite(val) and val > 0:
                    rows.append((v, val))
        t = pd.DataFrame(rows, columns=["vid", "r"])
        g = t.groupby("vid").r.agg(["std", "mean", "size"])
        g = g[g["size"] >= 4]
        return float((g["std"] / g["mean"]).median()) if len(g) else None
    out["cv_tib_tmt_pred"] = cv_table(pred)
    out["cv_tib_tmt_gt"] = cv_table(gt)
    return out


def summarize_runs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        rows.append({"run": d.name, "knee_pck0.1": m["per_joint"]["knee"]["pck@0.1_diag"],
                     "hip_pck0.1": m["per_joint"]["hip"]["pck@0.1_diag"],
                     "knee_pck0.05": m["per_joint"]["knee"]["pck@0.05_diag"],
                     "itj_pck0.1": m["per_joint"]["ankle"]["pck@0.1_diag"],
                     "visible_rmse_px": m["visible_rmse_px"], "cv_pred": m["cv_tib_tmt_pred"], "cv_gt": m["cv_tib_tmt_gt"],
                     "n_test": m["n_test_frames"]})
    return pd.DataFrame(rows)
