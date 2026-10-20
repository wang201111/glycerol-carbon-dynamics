"""Main-model training, evaluation and saved-model prediction."""

from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/main"
DEFAULT_INPUT = ROOT / "data/glycerol.csv"
DEVICE = "auto"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from src.models.carbon_dynamics import CarbonDynamics, CarbonDynamicsConfig, predict_ensemble
from src.models.utils_glycerol import StudyDataset, TARGETS


def regression_metrics(truth, prediction, minimum_variance=1e-16):
    observed = np.isfinite(truth)
    y, p = np.asarray(truth)[observed], np.asarray(prediction)[observed]
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite predictions for observed labels")
    if len(y) == 0:
        return dict(n=0, r2=np.nan, mae=np.nan, rmse=np.nan)
    residual = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    return dict(
        n=len(y),
        r2=(
            1 - np.sum(residual**2) / denominator
            if len(y) > 1 and denominator > minimum_variance
            else np.nan
        ),
        mae=100 * np.abs(residual).mean(),
        rmse=100 * np.sqrt(np.mean(residual**2)),
    )


def score_partition(truth, prediction, settings):
    rows = []
    for j, name in enumerate(TARGETS):
        rows.append(
            dict(
                target=name,
                **regression_metrics(truth[:, j], prediction[:, j], settings["r2_denominator_min"]),
            )
        )
    valid = (
        np.isfinite(truth[:, 0])
        & np.isfinite(truth[:, 1])
        & (truth[:, 1] > settings["selection_conversion_threshold"])
    )
    observed_s = truth[valid, 0] / truth[valid, 1]
    predicted_s = np.clip(
        prediction[valid, 0]
        / np.maximum(prediction[valid, 1], settings["ratio_denominator_floor"]),
        *settings["ratio_clip"],
    )
    rows.append(dict(target="selectivity", **regression_metrics(observed_s, predicted_s)))
    rows.append(
        dict(
            target="species_pooled",
            **regression_metrics(truth[:, 2:].ravel(), prediction[:, 2:].ravel()),
        )
    )
    return rows


def train(output, config, splits, seeds, device, smoke_test):
    torch.set_num_threads(1)
    started = time.perf_counter()
    predictions, metrics = [], []
    for split in splits:
        dataset = StudyDataset.load(split)
        members = []
        # Full-batch fits share one accelerator; each completed fit is saved before evaluation.
        for seed in seeds:
            model = CarbonDynamics(split=split, seed=seed, device=device, config=config)
            model.fit(dataset, max_updates=3 if smoke_test else None)
            model.save(output / f"s{split}_i{seed}.pt")
            members.append(model.predict_arrays(dataset.x, dataset.conditions)[0])
            print(f"Saved split={split}, initialization={seed}, device={model.device}", flush=True)
        prediction = np.mean(members, axis=0)
        frame = pd.DataFrame(
            dict(
                split=split,
                row_index=dataset.design["row_index"],
                partition=dataset.partition_names(),
            )
        )
        for j, name in enumerate(TARGETS):
            frame["observed_" + name] = 100 * dataset.targets[:, j]
            frame["predicted_" + name] = 100 * prediction[:, j]
        predictions.append(frame)
        for name, key in (("train", "tr"), ("validation", "va"), ("test", "te")):
            index = dataset.design[key]
            metrics.extend(
                dict(split=split, partition=name, **row)
                for row in score_partition(
                    dataset.targets[index], prediction[index], model.settings["evaluation"]
                )
            )
    per_split = pd.DataFrame(metrics)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    per_split.to_csv(output / "metrics.csv", index=False)
    if len(splits) > 1:
        summary = per_split.groupby(["partition", "target"])[["r2", "mae", "rmse"]].agg(
            ["mean", "std"]
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(output / "summary.csv")
    print(
        f"Completed {len(splits)*len(seeds)} fits in {time.perf_counter()-started:.1f} s; limited-budget run={smoke_test}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--split", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, nargs="+")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    config = CarbonDynamicsConfig()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    if args.mode == "predict":
        if not args.checkpoint:
            raise ValueError("Prediction requires one or more saved checkpoints")
        models = [CarbonDynamics.load(path, device=args.device) for path in args.checkpoint]
        prediction = predict_ensemble(models, pd.read_csv(args.input))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        prediction.to_csv(args.output_dir / "predictions.csv", index=False)
        print(f"Predicted {len(prediction)} records")
    else:
        splits = config.splits if args.split is None else (args.split,)
        seeds = config.initializations if args.seed is None else (args.seed,)
        if not set(splits).issubset(config.splits) or not set(seeds).issubset(
            config.initializations
        ):
            raise ValueError("Use the registered split and initialization identifiers")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        train(args.output_dir, config, splits, seeds, args.device, args.smoke_test)


if __name__ == "__main__":
    main()
