"""Frozen glycerol data and target-independent input transformations."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(root=None) -> dict:
    from .carbon_dynamics import CarbonDynamicsConfig

    return CarbonDynamicsConfig().to_settings()


TARGETS = ("c3", "conversion", "GLYD", "DHA", "GLYA", "TART", "LA")


TARGET_COLUMNS = (
    "C3_carbon_yield_pct",
    "conversion_for_aux_pct",
    "yieldC_GLYD_pct",
    "yieldC_DHA_pct",
    "yieldC_GLYA_pct",
    "yieldC_TART_pct",
    "yieldC_LA_pct",
)


@dataclass
class FrozenEncoder:
    base: dict
    extra: dict
    settings: dict

    @classmethod
    def load(cls, split: int, root: Path | str | None = None) -> "FrozenEncoder":
        root = project_root() if root is None else Path(root)
        config = load_config(root)
        if split not in config["splits"]:
            raise ValueError(f"Unknown frozen split: {split}")

        encoders = json.loads((root / "data/preprocessing.json").read_text(encoding="utf-8"))[
            str(split)
        ]
        return cls(encoders["base"], encoders["extra"], config["input"])

    def to_dict(self) -> dict:
        return dict(base=self.base, extra=self.extra, settings=self.settings)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Encode canonical input columns; response columns are never accessed.

        Physical conditions are T [K], oxygen pressure [bar], base/glycerol
        molar ratio, initial glycerol [mol/L], and time [h], in that order.
        """
        n = len(frame)
        if n == 0:
            raise ValueError("At least one input row is required")
        enc, settings = self.base, self.settings

        def numeric(column):
            if column not in frame:
                return np.full(n, np.nan)
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Infinite input in {column}")
            return values

        values = np.column_stack([numeric(col) for col in enc["numeric_columns"]])
        missing = np.isnan(values)
        filled = np.where(missing, np.asarray(enc["numeric_medians"])[None, :], values)
        with_indicators = np.column_stack(
            [filled, missing[:, enc["indicator_columns"]].astype(float)]
        )
        scaled = (with_indicators - np.asarray(enc["numeric_mean"])) / np.asarray(
            enc["numeric_scale"]
        )
        chunks = [scaled]
        for column, vocabulary in zip(enc["categorical_columns"], enc["categories"]):
            series = frame[column] if column in frame else pd.Series([np.nan] * n)
            categories = series.fillna(settings["legacy_missing_category"]).to_numpy()
            chunks.append(
                np.column_stack([categories == category for category in vocabulary]).astype(float)
            )
        base = np.column_stack(chunks)
        slots = settings["frozen_feature_slots"]
        if base.shape[1] > slots:
            raise ValueError("Encoded base features exceed the reserved slots")
        x = np.zeros((n, slots + settings["condition_slot_count"] + settings["time_slot_count"]))
        x[:, : base.shape[1]] = base
        ag = numeric(settings["ag_field"])
        absent = np.isnan(ag)
        ag_filled = np.where(absent, self.extra["ag"]["median"], ag)
        extra = [
            (ag_filled - self.extra["ag"]["mean"]) / self.extra["ag"]["scale"],
            absent.astype(float),
        ]
        for column in settings["extra_fields"]:
            series = frame[column] if column in frame else pd.Series([np.nan] * n)
            categories = series.fillna(settings["extra_missing_category"]).astype(str).to_numpy()
            extra.extend(
                (categories == category).astype(float)
                for category in self.extra["vocabulary"][column]
            )
        extra = np.column_stack(extra)
        width = settings["extra_width"]
        if extra.shape[1] > width:
            raise ValueError("Extra feature width exceeds the frozen schema")
        x = np.column_stack([x, np.pad(extra, ((0, 0), (0, width - extra.shape[1])))])

        pressure = numeric("P_eff")
        pressure = np.where(np.isnan(pressure), numeric("P_O2_bar"), pressure)
        mode = (
            frame["oxidant_delivery_mode"].fillna("").astype(str).to_numpy()
            if "oxidant_delivery_mode" in frame
            else np.full(n, "")
        )
        pressure = np.where(
            np.isnan(pressure) & np.isin(mode, settings["atmospheric_delivery_modes"]),
            settings["legacy_pressure_fill_bar"],
            pressure,
        )
        medians = enc["condition_medians"]
        columns = [
            ("T_K", numeric("T_K")),
            ("P_eff", pressure),
            ("base_GLY_ratio", numeric("base_GLY_ratio")),
            ("C_GLY_0_mol_L", numeric("C_GLY_0_mol_L")),
        ]
        conditions = [
            np.where(np.isnan(value), medians[column], value) for column, value in columns
        ]
        time_min = numeric("t_min_v9")
        if not np.isfinite(time_min).all() or (time_min < 0).any():
            raise ValueError("t_min_v9 must contain finite, nonnegative times in minutes")
        c = np.column_stack([*conditions, time_min / 60.0])
        if not np.isfinite(c).all() or (c[:, 0] <= 0).any():
            raise ValueError("Conditions must be finite and temperature must be positive")
        if (c[:, 1:] < 0).any():
            raise ValueError("Pressure, concentration, base ratio and time must be nonnegative")
        x[:, slots] = (c[:, 0] - settings["temperature_center_K"]) / settings["temperature_scale_K"]
        x[:, slots + 1] = np.log(np.maximum(c[:, 1], settings["pressure_floor_bar"]))
        x[:, slots + 2] = np.log1p(np.maximum(c[:, 2], settings["nonnegative_log_floor"]))
        x[:, slots + 3] = np.log1p(np.maximum(c[:, 3], settings["nonnegative_log_floor"]))
        if not np.isfinite(x).all():
            raise ValueError("Preprocessing produced nonfinite inputs")
        return x, c


@dataclass
class StudyDataset:
    split: int
    frame: pd.DataFrame
    design: dict[str, np.ndarray]
    encoder: FrozenEncoder
    x: np.ndarray
    conditions: np.ndarray
    targets: np.ndarray

    @classmethod
    def load(cls, split: int = 101, root: Path | str | None = None) -> "StudyDataset":
        root = project_root() if root is None else Path(root)
        encoder = FrozenEncoder.load(split, root)
        prefix = f"s{split}__"
        with np.load(root / "data/fixed_splits.npz", allow_pickle=False) as archive:
            design = {
                key[len(prefix) :]: archive[key].copy()
                for key in archive.files
                if key.startswith(prefix)
            }
        frame = (
            pd.read_csv(root / "data/glycerol.csv")
            .set_index("row_index", drop=False)
            .loc[design["row_index"]]
            .reset_index(drop=True)
        )
        x, c = encoder.transform(frame)
        np.testing.assert_allclose(x[:, : design["X"].shape[1]], design["X"], atol=1e-12, rtol=0)
        expected = np.column_stack([design[f"cond_{key}"] for key in ("T", "P", "B", "C0", "th")])
        np.testing.assert_allclose(c, expected, atol=1e-12, rtol=0)
        indices = np.concatenate([design[key] for key in ("tr", "va", "te")])
        if len(indices) != len(frame) or not np.array_equal(
            np.sort(indices), np.arange(len(frame))
        ):
            raise ValueError("Frozen partitions must be disjoint and cover the data")
        targets = np.column_stack([design["y_c3"], design["y_conv"], design["y_sp"]])
        np.testing.assert_allclose(
            frame[list(TARGET_COLUMNS)].to_numpy(float) / 100,
            targets,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        )
        return cls(split, frame, design, encoder, x, c, targets)

    def partition_names(self) -> np.ndarray:
        names = np.full(len(self.frame), "train", dtype="U10")
        names[self.design["va"]] = "validation"
        names[self.design["te"]] = "test"
        return names
