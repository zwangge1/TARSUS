"""W1 task 1b: where along the tarsometatarsus do SuperAnimal-Bird's leg points sit?

For every labeled frame and each side, the human-labeled itj (Duong `ankle`) and tmp
(Duong `foot`) define the tarsometatarsus axis. Each same-side SA point p is expressed as
  t    = (p - itj) . (tmp - itj) / |tmp - itj|^2      0 = itj, 1 = tmp, <0 toward the knee
  perp = |cross(tmp - itj, p - itj)| / |tmp - itj|^2  distance off the axis, in TMT lengths
Inputs are the outputs of w1_superbird_match.py (no new inference).
Usage: python scripts/w1_tmt_axis.py --out outputs/w1_match
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from w1_superbird_match import load_labels, md  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SA_CFG = REPO / "third_party/DeepLabCut/deeplabcut/modelzoo/project_configs/superanimal_bird.yaml"
BINS = [(-np.inf, -0.25, "above itj (toward knee)"), (-0.25, 0.25, "at itj"), (0.25, 0.75, "mid TMT"),
        (0.75, 1.25, "at tmp"), (1.25, np.inf, "beyond tmp (toes)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-tmt-px", type=float, default=5.0)
    a = ap.parse_args()

    sa_names = yaml.safe_load(SA_CFG.read_text())["bodyparts"]
    preds = json.loads((a.out / "sa_predictions.json").read_text())
    labels = load_labels(a.out / "project")
    labels = labels[~labels.index.duplicated()]

    rows = []
    for img, p in preds.items():
        if img not in labels.index or len(p["bodyparts"]) == 0:
            continue
        kp = np.asarray(p["bodyparts"])[0]
        lab = labels.loc[img].unstack()[["x", "y"]].astype(float)
        for side in ("left", "right"):
            itj, tmp = lab.loc[f"{side}_ankle"].to_numpy(), lab.loc[f"{side}_foot"].to_numpy()
            if not np.isfinite(np.r_[itj, tmp]).all():
                continue
            v = tmp - itj
            L2 = float(v @ v)
            if L2 < a.min_tmt_px ** 2:
                continue
            for part in ("tibia", "leg", "foot"):
                x, y, sc = kp[sa_names.index(f"{side}_{part}")]
                w = np.array([x, y]) - itj
                rows.append(dict(image=img, species=Path(img).parent.name.split("_")[0], side=side, sa_part=part,
                                 score=float(sc), t=float(w @ v / L2), perp=float(abs(v[0] * w[1] - v[1] * w[0]) / L2),
                                 tmt_px=float(np.sqrt(L2))))
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "tmt_axis.csv", index=False)

    def summarize(df):
        out = []
        for part, g in df.groupby("sa_part"):
            r = {"sa_part": part, "n": len(g), "t_median": g.t.median(), "t_q25": g.t.quantile(.25),
                 "t_q75": g.t.quantile(.75), "perp_median": g.perp.median()}
            for lo, hi, name in BINS:
                r[name] = ((g.t > lo) & (g.t <= hi)).mean()
            out.append(r)
        return pd.DataFrame(out).set_index("sa_part").reindex(["tibia", "leg", "foot"]).round(3)

    with (a.out / "tmt_axis_summary.md").open("w", encoding="utf-8") as f:
        f.write("# W1b: SA leg points along the tarsometatarsus axis (t: 0 = itj, 1 = tmp; perp in TMT lengths)\n\n")
        f.write(f"legs: {d.groupby(['image', 'side']).ngroups}, frames: {d.image.nunique()}, "
                f"median TMT length {d.tmt_px.median():.1f} px\n\n")
        f.write("## All predictions\n\n" + md(summarize(d)) + "\n\n")
        f.write("## SA score >= 0.5 only\n\n" + md(summarize(d[d.score >= 0.5])) + "\n\n")
        f.write("## Median t by species\n\n" + md(d.pivot_table(index="sa_part", columns="species", values="t",
                                                               aggfunc="median").round(3)) + "\n")
    print((a.out / "tmt_axis_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
