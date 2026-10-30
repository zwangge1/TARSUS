"""Build prior_v0.json: species/genus/family/global priors on hindlimb bone ratios.

Quantity: r = [log(fem/tmt), log(tib/tmt)]  (femur, tibiotarsus, tarsometatarsus; 3D bone lengths)
Each entry: mu (2,), Sigma (2,2), level, n_species, n_specimens, sources.

Sources (v0, Duong taxa only + a global entry for experiment group B):
  Eliason 2023 S1_limb_data.csv  specimen-level rows (fem/tbt/tmt)     -> within- and between-species spread
  Watanabe 2017 Anatidae           species means (fem/tib/tmt, n)        -> extra species means for Anatidae
Sigma rules:
  species  : within-species covariance (own specimens if >=3, else pooled within-species cov of its family)
  genus/fam: between-species covariance of species means in the group + pooled within-species cov
  global   : same, over all species in Eliason + Watanabe
Also writes species_map.csv (Duong video folder -> prior key) and proj_diag.json (2D label ratios vs prior,
used later to set Sigma_proj; nothing from it is applied in v0).
Usage: python prior/build_prior_v0.py --evowm-raw <EvoWM/data/raw> --duong <duong_data> --out prior/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {  # Duong video prefix -> prior key (genus level where the species is unknown)
    "Goose": "group:Anserini_geese",   # Anser + Branta
    "Swan": "genus:Cygnus",
    "Pelican": "genus:Pelecanus",
}
DUCK_SPECIES = {"Mallard": "Anas platyrhynchos", "Blue-winged Teal": "Spatula discors"}


def ratios(df, fem, tib, tmt):
    return np.c_[np.log(df[fem] / df[tmt]), np.log(df[tib] / df[tmt])]


def load(raw: Path) -> pd.DataFrame:
    e = pd.read_csv(raw / "eliason2023" / "S1_limb_data.csv", encoding="utf-8-sig")
    e = e.rename(columns={"Taxon": "species", "Family": "family"})
    e = e.dropna(subset=["fem_leng_mm", "tbt_leng_mm", "tmt_leng_mm"])
    e = e[(e[["fem_leng_mm", "tbt_leng_mm", "tmt_leng_mm"]] > 0).all(axis=1)]
    r = ratios(e, "fem_leng_mm", "tbt_leng_mm", "tmt_leng_mm")
    spec = pd.DataFrame({"species": e.species.str.strip(), "family": e.family, "r1": r[:, 0], "r2": r[:, 1],
                         "source": "Eliason2023", "kind": "specimen"})
    w = pd.read_csv(raw / "watanabe2017" / "watanabe2017_anatidae.csv", encoding="utf-8-sig")
    rw = ratios(w, "fem_mm", "tib_mm", "tmt_mm")
    wat = pd.DataFrame({"species": w["species"].str.strip(), "family": "Anatidae", "r1": rw[:, 0], "r2": rw[:, 1],
                        "source": "Watanabe2017", "kind": "species_mean"})
    d = pd.concat([spec, wat], ignore_index=True)
    d["genus"] = d.species.str.split().str[0]
    return d


def pooled_within(d: pd.DataFrame) -> np.ndarray:
    s = d[d.kind == "specimen"]
    res = []
    for _, g in s.groupby("species"):
        if len(g) >= 2:
            x = g[["r1", "r2"]].to_numpy()
            res.append(x - x.mean(0))
    x = np.vstack(res)
    return x.T @ x / max(len(x) - len(res), 1)


def species_means(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("species").agg(r1=("r1", "mean"), r2=("r2", "mean"), family=("family", "first"),
                                    genus=("genus", "first"), n=("r1", "size"),
                                    sources=("source", lambda s: sorted(set(s))))


def entry(level, mu, cov, n_sp, n_spec, sources):
    return {"level": level, "mu": np.round(mu, 5).tolist(), "Sigma": np.round(cov, 6).tolist(),
            "n_species": int(n_sp), "n_specimens": int(n_spec), "sources": sorted(set(sources))}


def group_entry(level, d, sm, mask_sp, within):
    m = sm[mask_sp]
    x = m[["r1", "r2"]].to_numpy()
    between = np.cov(x.T) if len(x) >= 3 else np.zeros((2, 2))
    srcs = [s for l in m.sources for s in l]
    return entry(level, x.mean(0), between + within, len(m), d.species.isin(m.index).sum(), srcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evowm-raw", type=Path, required=True)
    ap.add_argument("--duong", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    d = load(a.evowm_raw)
    sm = species_means(d)
    within_all = pooled_within(d)
    within_fam = {f: pooled_within(d[d.family == f]) for f in ["Anatidae", "Pelecanidae"]}
    prior = {}

    # species entries for the duck species
    for sp in DUCK_SPECIES.values():
        if sp not in sm.index:  # e.g. Spatula discors: no skeletal data in v0 sources -> genus:Anas
            continue
        g = d[(d.species == sp) & (d.kind == "specimen")]
        x = g[["r1", "r2"]].to_numpy()
        cov = np.cov(x.T) if len(x) >= 3 else within_fam["Anatidae"]
        prior[f"species:{sp}"] = entry("species", sm.loc[sp, ["r1", "r2"]].to_numpy(float), cov,
                                       1, (d.species == sp).sum(), sm.loc[sp, "sources"])
    # genus / group / family / global
    prior["genus:Anas"] = group_entry("genus", d, sm, sm.genus == "Anas", within_fam["Anatidae"])  # dabbling ducks, fallback for teal
    prior["group:Anserini_geese"] = group_entry("group", d, sm, sm.genus.isin(["Anser", "Branta"]), within_fam["Anatidae"])
    prior["genus:Cygnus"] = group_entry("genus", d, sm, sm.genus == "Cygnus", within_fam["Anatidae"])
    prior["genus:Pelecanus"] = group_entry("genus", d, sm, sm.genus == "Pelecanus", within_fam["Pelecanidae"])
    prior["family:Anatidae"] = group_entry("family", d, sm, sm.family == "Anatidae", within_fam["Anatidae"])
    prior["family:Pelecanidae"] = group_entry("family", d, sm, sm.family == "Pelecanidae", within_fam["Pelecanidae"])
    prior["global:Aves"] = group_entry("global", d, sm, sm.index.notna(), within_all)

    meta = {"version": "v0", "date": "2026-10-30",
            "quantity": "r = [log(femur/tarsometatarsus), log(tibiotarsus/tarsometatarsus)], 3D bone lengths",
            "joints": {"femur": "hip-knee", "tibiotarsus": "knee-ankle(itj)", "tarsometatarsus": "ankle(itj)-foot(tmp)"}}
    (a.out / "prior_v0.json").write_text(json.dumps({"meta": meta, "priors": prior}, indent=1), encoding="utf-8")

    # species_map.csv: Duong labeled-data folder -> prior key
    xl = pd.read_excel(a.duong / "Raw datasets" / "Videos.xlsx", sheet_name="Duck")
    duck_id = {int(i): DUCK_SPECIES.get(s, "Anas platyrhynchos") for i, s in zip(xl.ID, xl.Species) if pd.notna(i)}
    proj = a.duong / "deeplabcut" / "bird-landing-Duong-2025-06-29"
    rows = []
    for folder in sorted(p.name for p in (proj / "labeled-data").iterdir() if p.is_dir()):
        pre = folder.split("_")[0]
        num = re.match(r"[A-Za-z]+_(\d+)", folder)
        if pre == "Duck":
            key = f"species:{duck_id.get(int(num.group(1)), 'Anas platyrhynchos')}" if num else "species:Anas platyrhynchos"
        else:
            key = GROUPS.get(pre)
        rows.append({"folder": folder, "video_id": f"{pre}_{num.group(1)}" if num else folder, "prior_key": key})
    for r in rows:
        if r["prior_key"] not in prior:
            r["prior_key"] = "genus:Anas"
    pd.DataFrame(rows).to_csv(a.out / "species_map.csv", index=False)

    # proj_diag.json: 2D human-label ratios vs the 3D prior (diagnostic only)
    smap = pd.DataFrame(rows).set_index("folder").prior_key
    diag = {}
    for csv in sorted((proj / "labeled-data").glob("*/CollectedData_*.csv")):
        lab = pd.read_csv(csv, header=[0, 1, 2], index_col=[0, 1, 2])
        lab.columns = lab.columns.droplevel(0)
        key = smap[csv.parent.name]
        for side in ("left", "right"):
            try:
                P = {j: lab[f"{side}_{j}"][["x", "y"]].to_numpy(float) for j in ("hip", "knee", "ankle", "foot")}
            except KeyError:
                continue
            fem = np.linalg.norm(P["hip"] - P["knee"], axis=1)
            tib = np.linalg.norm(P["knee"] - P["ankle"], axis=1)
            tmt = np.linalg.norm(P["ankle"] - P["foot"], axis=1)
            ok = np.isfinite(fem + tib + tmt) & (fem > 2) & (tib > 2) & (tmt > 2)
            diag.setdefault(key, []).extend(np.c_[np.log(fem[ok] / tmt[ok]), np.log(tib[ok] / tmt[ok])].tolist())
    out = {}
    for key, v in diag.items():
        x = np.asarray(v)
        mu = np.asarray(prior[key]["mu"])
        out[key] = {"n_legs": len(x), "mean_2d": np.round(x.mean(0), 4).tolist(), "prior_mu_3d": mu.tolist(),
                    "shift_2d_minus_3d": np.round(x.mean(0) - mu, 4).tolist(),
                    "cov_2d": np.round(np.cov(x.T), 5).tolist(), "prior_Sigma": prior[key]["Sigma"]}
    (a.out / "proj_diag.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    for k, v in prior.items():
        print(f"{k:32s} n_sp={v['n_species']:4d} n_spec={v['n_specimens']:5d} mu={v['mu']} sd={np.sqrt(np.diag(v['Sigma'])).round(3).tolist()}")
    print("\n2D label check (shift = mean 2D ratio - 3D prior mean; negative = foreshortened):")
    for k, v in out.items():
        print(f"{k:32s} legs={v['n_legs']:4d} shift={v['shift_2d_minus_3d']} sd2d={np.sqrt(np.diag(v['cov_2d'])).round(3).tolist()}")


if __name__ == "__main__":
    main()
