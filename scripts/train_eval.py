"""Train + evaluate one TARSUS run (group x species-held-out fold x seed) with official DLC 3 APIs.

Group A (this version): official top_down_resnet_50, SuperAnimal-Bird backbone + detector init
(with_decoder=False: the SA head has no knee/hip, so every head channel is re-initialised and
learned from Duong's labels), trained on three species, tested on the held-out species.
Groups B/C/D will add the prior head via --head-config (not implemented yet).

Shuffle id = GROUP_BASE[group] + FOLD_BASE[fold] + seed, so every run has its own DLC shuffle.
Outputs: outputs/runs/<group>_<fold>_s<seed>/{metrics.json, run_info.json}; the DLC model and
evaluation files stay in the working project under dlc-models-pytorch / evaluation-results-pytorch.

Usage: python scripts/train_eval.py --group A --fold Duck --seed 0 [--epochs 200] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "outputs" / "dlc_project"
GROUP_BASE = {"A": 100, "B": 200, "C": 300, "D": 400}
FOLD_BASE = {"Duck": 0, "Goose": 10, "Pelican": 20, "Swan": 30}
SA = "superanimal_bird"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUP_BASE), required=True)
    ap.add_argument("--fold", choices=list(FOLD_BASE), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--detector-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-train", action="store_true", help="only evaluate an existing shuffle")
    a = ap.parse_args()
    if a.group != "A":
        raise SystemExit("groups B/C/D need the prior head (M1-M5), not implemented yet")

    import deeplabcut
    from deeplabcut.core.engine import Engine
    from deeplabcut.core.weight_init import WeightInitialization
    from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
    from deeplabcut.utils import auxiliaryfunctions as af

    cfg_path = PROJECT / "config.yaml"
    split = json.loads((PROJECT / "split.json").read_text())[a.fold]
    index_map = pd.read_csv(PROJECT / "index_map.csv")
    shuffle = GROUP_BASE[a.group] + FOLD_BASE[a.fold] + a.seed
    run_dir = REPO / "outputs" / "runs" / f"{a.group}_{a.fold}_s{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # make sure this fold's training fraction is listed in config.yaml (DLC locates model
    # folders through TrainingFraction[trainingsetindex])
    cfg = af.read_config(str(cfg_path))
    frac = split["fraction"]
    fractions = [round(float(f), 2) for f in cfg["TrainingFraction"]]
    if frac not in fractions:
        fractions.append(frac)
        cfg["TrainingFraction"] = fractions
        af.write_config(str(cfg_path), cfg)
    trainset_index = fractions.index(frac)

    t0 = time.time()
    if not a.skip_train:
        weight_init = WeightInitialization(
            snapshot_path=get_super_animal_snapshot_path(SA, "resnet_50"),
            detector_snapshot_path=get_super_animal_snapshot_path(SA, "fasterrcnn_mobilenet_v3_large_fpn"),
            dataset=SA, with_decoder=False,
        )
        deeplabcut.create_training_dataset(
            str(cfg_path), Shuffles=[shuffle], trainIndices=[split["train_idx"]], testIndices=[split["test_idx"]],
            net_type="top_down_resnet_50", detector_type="fasterrcnn_mobilenet_v3_large_fpn",
            weight_init=weight_init, engine=Engine.PYTORCH, userfeedback=False,
        )
        # seed + epochs into the shuffle's pytorch_config.yaml (official training reads these)
        loader = DLCLoader(cfg_path, trainset_index=trainset_index, shuffle=shuffle)
        pcfg_path = Path(loader.model_config_path)
        pcfg = yaml.safe_load(pcfg_path.read_text())
        pcfg["train_settings"]["seed"] = 42 + a.seed
        pcfg["train_settings"]["epochs"] = a.epochs
        pcfg["train_settings"]["batch_size"] = a.batch_size
        pcfg["train_settings"]["dataloader_workers"] = 4
        if "detector" in pcfg:
            pcfg["detector"]["train_settings"]["epochs"] = a.detector_epochs
        pcfg_path.write_text(yaml.safe_dump(pcfg, sort_keys=False))
        deeplabcut.train_network(str(cfg_path), shuffle=shuffle, trainingsetindex=trainset_index,
                                 device=a.device, epochs=a.epochs, detector_epochs=a.detector_epochs,
                                 batch_size=a.batch_size)
    t_train = time.time() - t0

    deeplabcut.evaluate_network(str(cfg_path), shuffles=[shuffle], trainingsetindex=trainset_index,
                                snapshotindex=-1, device=a.device, per_keypoint_evaluation=True,
                                detector_snapshot_index=-1)

    loader = DLCLoader(cfg_path, trainset_index=trainset_index, shuffle=shuffle)
    h5s = sorted(loader.evaluation_folder.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    pred = pd.read_hdf(h5s[-1])
    gt = loader.df
    m = metrics.compute(gt, pred, split["test_idx"], index_map)
    m.update({"group": a.group, "fold": a.fold, "seed": a.seed, "shuffle": shuffle, "epochs": a.epochs,
              "train_minutes": round(t_train / 60, 1), "prediction_file": str(h5s[-1])})
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    k = m["per_joint"]
    print(f"\n== {run_dir.name}: knee PCK@0.1 {k['knee']['pck@0.1_diag']:.3f}  hip {k['hip']['pck@0.1_diag']:.3f}  "
          f"itj {k['ankle']['pck@0.1_diag']:.3f}  visible RMSE {m['visible_rmse_px']:.1f}px  "
          f"CV pred {m['cv_tib_tmt_pred']}  CV gt {m['cv_tib_tmt_gt']}  ({m['train_minutes']} min)")


if __name__ == "__main__":
    main()
