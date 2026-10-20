"""Conditional neural dynamics constrained by a shared carbon-transfer structure."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional
from .utils_glycerol import FrozenEncoder, StudyDataset, project_root


@dataclass
class CarbonDynamicsConfig:
    """Main-model settings used by the paper experiment."""

    splits: tuple[int, ...] = (101, 123, 314, 73, 202)
    initializations: tuple[int, ...] = (5, 11)
    hidden_width: int = 96
    clock_latent_width: int = 3
    log_rate_bounds: tuple[float, float] = (-9.0, 5.0)
    learning_rate: float = 0.003
    weight_decay: float = 1e-05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-08
    gradient_clip: float = 5.0
    max_updates: int = 1096
    check_every: int = 5
    patience_updates: int = 180
    min_improvement: float = 1e-07
    loss_weights: tuple[float, ...] = (1.0, 0.3, 0.3, 0.3)
    species_variance_floor_std: float = 0.05
    ratio_floor: float = 1e-06
    conversion_threshold: float = 0.05
    ratio_clip: tuple[float, float] = (0.0, 1.5)
    initial_exponent: float = 1.0
    initial_temperature_coefficient: float = 0.0
    material_scale: float = 2.0

    def to_settings(self) -> dict:
        return {
            "schema_version": 1,
            "default_model": "hierarchical_31",
            "splits": self.splits,
            "initializations": self.initializations,
            "input": {
                "ag_field": "Ag_per_phosphomolybdate_nominal",
                "atmospheric_delivery_modes": [
                    "continuous_flow",
                    "flowing",
                    "continuous_bubbling",
                    "atmospheric",
                ],
                "condition_slot_count": 4,
                "extra_fields": ["solvent_mode", "pretreatment_completed"],
                "extra_missing_category": "__MISSING__",
                "extra_width": 7,
                "frozen_feature_slots": 120,
                "legacy_missing_category": "UNKNOWN",
                "legacy_pressure_fill_bar": 1.01325,
                "nonnegative_log_floor": 0.0,
                "pressure_floor_bar": 0.1,
                "temperature_center_K": 343.15,
                "temperature_scale_K": 30.0,
                "time_slot_count": 1,
            },
            "network": {
                "clock_latent_width": self.clock_latent_width,
                "hidden_width": self.hidden_width,
                "source_log_rate_bounds": self.log_rate_bounds,
            },
            "loss": {
                "ratio_denominator_floor": self.ratio_floor,
                "selection_conversion_threshold": self.conversion_threshold,
                "species_variance_floor_std": self.species_variance_floor_std,
                "weights": self.loss_weights,
            },
            "optimizer": {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "gradient_clip": self.gradient_clip,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "training": {
                "check_every": self.check_every,
                "max_updates": self.max_updates,
                "min_improvement": self.min_improvement,
                "patience_updates": self.patience_updates,
            },
            "evaluation": {
                "comparison_tolerance": 1e-12,
                "constraint_slack": 1e-09,
                "r2_denominator_min": 1e-16,
                "ratio_clip": self.ratio_clip,
                "ratio_denominator_floor": self.ratio_floor,
                "selection_conversion_threshold": self.conversion_threshold,
            },
            "initialization": {"legacy_discard_normal_blocks": 2},
            "power_clock": {
                "initial_exponent": self.initial_exponent,
                "initial_temperature_coefficient": self.initial_temperature_coefficient,
                "material_scale": self.material_scale,
            },
            "models": {
                "hierarchical_31": {
                    "states": ["G", "GLYD", "DHA", "GLYA", "TART", "LA", "X"],
                    "edges": [(0, j) for j in range(1, 7)]
                    + [(i, j) for i in range(1, 6) for j in range(1, 7) if i != j],
                    "allocation": "hierarchical",
                    "supervision": "species",
                    "match_initial_target_share": False,
                }
            },
        }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    states: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    allocation: str
    supervision: str
    match_initial_target_share: bool = False

    @classmethod
    def from_study(cls, config: dict, name: str | None = None) -> "ModelConfig":
        name = config["default_model"] if name is None else name
        entry = copy.deepcopy(config["models"][name])
        entry.pop("source_id", None)
        entry["states"] = tuple(entry["states"])
        entry["edges"] = tuple(tuple(edge) for edge in entry["edges"])
        result = cls(name=name, **entry)
        n = len(result.states)
        if result.allocation not in ("softmax", "sparsemax", "hierarchical"):
            raise ValueError("Unknown allocation rule")
        if result.supervision not in ("total", "species"):
            raise ValueError("Unknown supervision rule")
        if result.states[0] != "G" or result.states[-1] != "X":
            raise ValueError("The first and last states must be G and X")
        if not result.edges or len(result.edges) != len(set(result.edges)):
            raise ValueError("Edges must be nonempty and unique")
        if not all(0 <= a < n - 1 and 0 < b < n and a != b for a, b in result.edges):
            raise ValueError("Invalid carbon-transfer edge")
        return result


PARAMETER_NAMES = (
    "W",
    "b",
    "Wz",
    "bz",
    "activity_W",
    "activity_bias",
    "activity_base",
    "activity_Eraw",
    "source_W",
    "source_bias",
    "branch_W",
    "branch_bias",
)


def sparsemax(logits: torch.Tensor) -> torch.Tensor:
    ordered = torch.sort(logits, dim=-1, descending=True).values
    cumulative = torch.cumsum(ordered, dim=-1)
    rank = torch.arange(1, logits.shape[-1] + 1, dtype=logits.dtype, device=logits.device)
    support_size = torch.sum((1 + rank * ordered) > cumulative, dim=-1, keepdim=True).clamp(min=1)
    threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(logits - threshold, min=0.0)


def initialize_parameters(model: ModelConfig, dataset, seed: int, settings: dict) -> dict:
    hidden = settings["network"]["hidden_width"]
    latent = settings["network"]["clock_latent_width"]
    edge_count = len(model.edges)
    rng = np.random.default_rng(seed)

    def linear(n_in, n_out):
        bound = 1 / np.sqrt(n_in)
        return rng.uniform(-bound, bound, (n_in, n_out)), rng.uniform(-bound, bound, (n_out,))

    weights, bias = linear(dataset.design["X"].shape[1], hidden)
    latent_weights, latent_bias = linear(hidden, latent)
    # Preserve the published initialization stream across connection configurations.
    for _ in range(settings["initialization"]["legacy_discard_normal_blocks"]):
        rng.normal(size=(latent, edge_count))
    clock_weights, clock_bias = linear(latent, 1)
    weights = np.pad(weights, ((0, dataset.x.shape[1] - weights.shape[0]), (0, 0)))
    donors = sorted({a for a, _ in model.edges})
    times = dataset.conditions[dataset.design["tr"], 4]
    times = times[times > 0]
    if not len(times):
        raise ValueError("Training data contain no positive reaction times")
    parameters = dict(
        W=weights,
        b=bias,
        Wz=latent_weights,
        bz=latent_bias,
        activity_W=np.zeros_like(clock_weights),
        activity_bias=np.zeros_like(clock_bias),
        activity_base=np.array([np.log(np.expm1(settings["power_clock"]["initial_exponent"]))]),
        activity_Eraw=np.array([settings["power_clock"]["initial_temperature_coefficient"]]),
        source_W=np.zeros((hidden, len(donors))),
        source_bias=np.full(len(donors), np.log(1 / np.median(times))),
        branch_W=np.zeros((hidden, edge_count)),
        branch_bias=np.zeros(edge_count),
    )
    if model.match_initial_target_share:
        target_outlets = sum(a == 0 and b != len(model.states) - 1 for a, b in model.edges)
        for j, (a, b) in enumerate(model.edges):
            if a == 0 and b != len(model.states) - 1:
                parameters["branch_bias"][j] = np.log(2 / target_outlets)
    return parameters


class CarbonTransferNetwork(nn.Module):
    """Column-generator evolution from an initial glycerol carbon fraction of one."""

    def __init__(self, config: ModelConfig, parameters: dict, settings: dict):
        super().__init__()
        self.config = config
        self.settings = settings
        n = len(config.states)
        self.donors = sorted({a for a, _ in config.edges})
        self.groups = [
            (
                [j for j, (a, _) in enumerate(config.edges) if a == donor],
                [b for a, b in config.edges if a == donor],
            )
            for donor in self.donors
        ]
        basis = np.zeros((len(config.edges), n, n), dtype=np.float64)
        for j, (donor, receiver) in enumerate(config.edges):
            basis[j, donor, donor] -= 1
            basis[j, receiver, donor] += 1
        self.register_buffer(
            "basis", torch.as_tensor(basis.reshape(len(config.edges), n * n)), persistent=False
        )
        self.register_buffer(
            "src_col",
            torch.tensor([self.donors.index(a) for a, _ in config.edges]),
            persistent=False,
        )
        self.register_buffer(
            "edge_donors", torch.tensor([a for a, _ in config.edges]), persistent=False
        )
        self._params = nn.ParameterDict(
            {
                name: nn.Parameter(torch.as_tensor(parameters[name], dtype=torch.float64).clone())
                for name in PARAMETER_NAMES
            }
        )

    def p(self, name):
        return self._params[name]

    def allocate(self, logits: torch.Tensor) -> torch.Tensor:
        sink = len(self.config.states) - 1
        result = torch.zeros_like(logits)
        for indices, destinations in self.groups:
            scores = logits[:, indices]
            if len(indices) == 1:
                probabilities = torch.ones_like(scores)
            elif self.config.allocation == "softmax":
                probabilities = torch.softmax(scores, dim=-1)
            elif self.config.allocation == "sparsemax":
                probabilities = sparsemax(scores)
            else:
                inside = [j for j, target in enumerate(destinations) if target != sink]
                outside = [j for j, target in enumerate(destinations) if target == sink]
                if not outside:
                    probabilities = sparsemax(scores)
                elif not inside:
                    probabilities = torch.ones_like(scores)
                else:
                    dense = torch.softmax(scores, dim=-1)
                    target_share = dense[:, inside].sum(dim=1, keepdim=True)
                    within = (
                        sparsemax(scores[:, inside])
                        if len(inside) > 1
                        else torch.ones_like(scores[:, inside])
                    )
                    columns = [None] * len(indices)
                    for local, j in enumerate(inside):
                        columns[j] = target_share * within[:, local : local + 1]
                    for j in outside:
                        columns[j] = dense[:, j : j + 1]
                    probabilities = torch.cat(columns, dim=1)
            index = torch.tensor(indices, dtype=torch.long, device=logits.device)
            result = result.index_copy(1, index, probabilities)
        return result

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, context: dict, *, detail=False):
        hidden = x @ self.p("W") + self.p("b")
        hidden = hidden * (0.5 * (1 + torch.tanh(0.5 * hidden)))
        latent = hidden @ self.p("Wz") + self.p("bz")
        lower, upper = self.settings["network"]["source_log_rate_bounds"]
        donor_rates = torch.exp(
            torch.clamp(hidden @ self.p("source_W") + self.p("source_bias"), lower, upper)
        )
        logits = hidden @ self.p("branch_W") + self.p("branch_bias")
        probabilities = self.allocate(logits)
        rates = probabilities * donor_rates.index_select(1, self.src_col)
        material = self.settings["power_clock"]["material_scale"] * torch.tanh(
            latent @ self.p("activity_W") + self.p("activity_bias")
        )
        temperature = (conditions[:, 0:1] - context["temperature_mean_K"]) / context[
            "temperature_scale_K"
        ]
        exponent = functional.softplus(
            self.p("activity_base") + material + self.p("activity_Eraw") * temperature
        )
        time = conditions[:, 4:5]
        reference_time = context["reference_time_h"]
        log_time = torch.log(
            torch.clamp(time / reference_time, min=torch.finfo(torch.float64).tiny)
        )
        effective_time = torch.where(
            time > 0, reference_time * torch.exp(exponent * log_time), torch.zeros_like(time)
        )
        n = len(self.config.states)
        generator = (rates @ self.basis).reshape(-1, n, n)
        states = torch.linalg.matrix_exp(generator * effective_time.unsqueeze(-1))[:, :, 0]
        conversion = 1 - states[:, 0]
        yield_c3 = states[:, 1:-1].sum(dim=1)
        outputs = torch.cat([yield_c3[:, None], conversion[:, None], states[:, 1:-1]], dim=1)
        if detail:
            return (
                outputs,
                states,
                dict(
                    rates=rates,
                    probabilities=probabilities,
                    donor_rates=donor_rates,
                    effective_time=effective_time,
                    exponent=exponent,
                    generator=generator,
                    logits=logits,
                ),
            )
        return outputs, states


def objective(prediction, targets, mask, species_weights, settings, include_species=True):
    error = (prediction[:, :2] - targets[:, :2]).square() * mask[:, :2]
    yield_loss = error[:, 0].sum() / mask[:, 0].sum().clamp(min=1)
    conversion_loss = error[:, 1].sum() / mask[:, 1].sum().clamp(min=1)
    if include_species:
        n = species_weights.numel()
        error = (prediction[:, 2 : 2 + n] - targets[:, 2 : 2 + n]).square() * mask[:, 2 : 2 + n]
        species_loss = (error * species_weights[None, :]).sum() / mask[:, 2 : 2 + n].sum().clamp(
            min=1
        )
    else:
        species_loss = torch.zeros_like(yield_loss)
    loss_config = settings["loss"]
    valid = (
        mask[:, 0]
        * mask[:, 1]
        * (targets[:, 1] > loss_config["selection_conversion_threshold"]).to(targets.dtype)
    )
    floor = loss_config["ratio_denominator_floor"]
    ratio_error = (
        prediction[:, 0] / prediction[:, 1].clamp(min=floor)
        - targets[:, 0] / targets[:, 1].clamp(min=floor)
    ).square()
    selectivity_loss = (ratio_error * valid).sum() / valid.sum().clamp(min=1)
    components = torch.stack([yield_loss, conversion_loss, species_loss, selectivity_loss])
    weights = list(loss_config["weights"])
    if not include_species:
        weights[2] = 0
    return (components * components.new_tensor(weights)).sum(), components


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(device)


class CarbonDynamics:
    """Conditional neural dynamics constrained by a shared carbon-transfer structure.

    ``fit`` uses a frozen StudyDataset. ``predict`` accepts raw canonical input
    columns. ``save`` embeds preprocessing and configuration so ``load`` does not
    depend on the study data directory.
    """

    def __init__(
        self, model="hierarchical_31", split=101, seed=5, device="auto", root=None, config=None
    ):
        self.root = project_root() if root is None else Path(root)
        self.settings = (
            CarbonDynamicsConfig().to_settings()
            if config is None
            else (
                config.to_settings()
                if isinstance(config, CarbonDynamicsConfig)
                else copy.deepcopy(config)
            )
        )
        self.config = ModelConfig.from_study(self.settings, model)
        if split not in self.settings["splits"] or seed not in self.settings["initializations"]:
            raise ValueError("Use registered split and initialization identifiers")
        self.split, self.seed = split, seed
        self.device = resolve_device(device)
        self.encoder = None
        self.context = None
        self.network = None
        self.history = []
        self.metadata = {}

    def fit(self, dataset: StudyDataset, *, max_updates: int | None = None) -> "CarbonDynamics":
        if dataset.split != self.split:
            raise ValueError("Dataset and model use different frozen splits")
        torch.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.encoder = dataset.encoder
        self.context = self.encoder.extra["clock_context"]
        parameters = initialize_parameters(self.config, dataset, self.seed, self.settings)
        self.network = CarbonTransferNetwork(self.config, parameters, self.settings).to(self.device)
        include_species = self.config.supervision == "species"
        truth = dataset.targets if include_species else dataset.targets[:, :2]
        valid = np.isfinite(truth)
        y = np.nan_to_num(truth, nan=0.0)
        tensors = [
            torch.as_tensor(value, dtype=torch.float64, device=self.device)
            for value in (dataset.x, dataset.conditions, y, valid.astype(float))
        ]
        x, c, y, mask = tensors
        train = torch.as_tensor(dataset.design["tr"], dtype=torch.long, device=self.device)
        validation = torch.as_tensor(dataset.design["va"], dtype=torch.long, device=self.device)
        if include_species:
            observed = truth[dataset.design["tr"], 2:]
            count = np.isfinite(observed).sum(axis=0)
            variance = np.maximum(
                np.nanvar(observed, axis=0),
                self.settings["loss"]["species_variance_floor_std"] ** 2,
            )
            if (count == 0).any() or not np.isfinite(variance).all():
                raise ValueError("Each supervised species needs observed training labels")
            raw = 1 / (count * variance)
            weights = raw * count.sum() / np.sum(raw * count)
        else:
            weights = np.zeros(0)
        species_weights = torch.as_tensor(weights, dtype=torch.float64, device=self.device)
        validation_weights = torch.ones_like(species_weights)
        opt = self.settings["optimizer"]
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=opt["learning_rate"],
            betas=(opt["beta1"], opt["beta2"]),
            eps=opt["epsilon"],
            weight_decay=opt["weight_decay"],
            foreach=False,
        )
        training = self.settings["training"]
        limit = training["max_updates"] if max_updates is None else max_updates
        if not isinstance(limit, int) or limit < 1 or limit > training["max_updates"]:
            raise ValueError("max_updates must be between 1 and the registered training budget")
        best_value, bad_updates, best_state, best_update = np.inf, 0, None, 0
        self.history = []
        start = time.perf_counter()
        for update in range(1, limit + 1):
            self.network.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = self.network(x[train], c[train], self.context)
            loss, _ = objective(
                prediction, y[train], mask[train], species_weights, self.settings, include_species
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training objective at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), opt["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if update == 1 or (update - 1) % training["check_every"] == 0:
                self.network.eval()
                with torch.no_grad():
                    prediction, _ = self.network(x[validation], c[validation], self.context)
                    val_loss, components = objective(
                        prediction,
                        y[validation],
                        mask[validation],
                        validation_weights,
                        self.settings,
                        include_species,
                    )
                value = float(val_loss)
                if not np.isfinite(value):
                    raise FloatingPointError(f"Nonfinite validation objective at update {update}")
                if value < best_value - training["min_improvement"]:
                    best_value, bad_updates, best_update = value, 0, update
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()
                    }
                else:
                    bad_updates += training["check_every"]
                self.history.append(
                    dict(
                        update=update,
                        train_objective=float(loss.detach()),
                        validation_objective=value,
                        val_yield=float(components[0]),
                        val_conversion=float(components[1]),
                        val_species=float(components[2]),
                        val_selectivity=float(components[3]),
                        best_update=best_update,
                    )
                )
                if update == 1 or update % 100 == 1:
                    print(
                        f"{self.config.name} split={self.split} init={self.seed} update={update} validation={value:.8f}",
                        flush=True,
                    )
                if bad_updates >= training["patience_updates"]:
                    break
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was obtained")
        self.network.load_state_dict(best_state, strict=True)
        self.metadata = dict(
            origin="training",
            split=self.split,
            initialization=self.seed,
            best_update=best_update,
            last_update=update,
            best_validation=best_value,
            budget_limited=limit < training["max_updates"],
            device=str(self.device),
            device_reason=(
                "CUDA available"
                if self.device.type == "cuda"
                else "explicit CPU or automatic CPU selection"
            ),
            seconds=time.perf_counter() - start,
            train_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(dataset.design["te"]),
            test_used_for_selection=False,
            species_weights=weights.tolist(),
        )
        return self

    def predict_arrays(self, x, conditions, *, detail=False, batch_size=256):
        if self.network is None or self.context is None:
            raise RuntimeError("Fit or load the model before prediction")
        x, conditions = np.asarray(x, dtype=float), np.asarray(conditions, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != self.network.p("W").shape[0]
            or conditions.shape != (len(x), 5)
        ):
            raise ValueError("Unexpected input dimensions")
        if (
            len(x) == 0
            or batch_size < 1
            or not np.isfinite(x).all()
            or not np.isfinite(conditions).all()
            or (conditions[:, 4] < 0).any()
        ):
            raise ValueError("Prediction requires nonempty finite inputs and nonnegative time")
        outputs, states, details = [], [], {}
        self.network.eval()
        with torch.no_grad():
            for begin in range(0, len(x), batch_size):
                stop = begin + batch_size
                tx = torch.as_tensor(x[begin:stop], dtype=torch.float64, device=self.device)
                tc = torch.as_tensor(
                    conditions[begin:stop], dtype=torch.float64, device=self.device
                )
                values = self.network(tx, tc, self.context, detail=detail)
                outputs.append(values[0].cpu().numpy())
                states.append(values[1].cpu().numpy())
                if detail:
                    for key, value in values[2].items():
                        details.setdefault(key, []).append(value.cpu().numpy())
        p, state = np.concatenate(outputs), np.concatenate(states)
        if not np.isfinite(p).all() or not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite model output")
        if state.min() < -1e-8 or np.max(np.abs(state.sum(axis=1) - 1)) > 1e-8:
            raise FloatingPointError("Carbon-state invariant failed")
        if detail:
            return p, state, {key: np.concatenate(value) for key, value in details.items()}
        return p, state

    def predict(self, frame: pd.DataFrame, *, batch_size=256) -> pd.DataFrame:
        if self.encoder is None:
            raise RuntimeError("Fit or load the model before prediction")
        x, c = self.encoder.transform(frame)
        prediction, states = self.predict_arrays(x, c, batch_size=batch_size)
        result = pd.DataFrame(index=frame.index)
        if "row_index" in frame:
            result["row_index"] = frame["row_index"]
        result["c3_yield_pct"] = 100 * prediction[:, 0]
        result["conversion_pct"] = 100 * prediction[:, 1]
        ratio = prediction[:, 0] / np.maximum(
            prediction[:, 1], self.settings["evaluation"]["ratio_denominator_floor"]
        )
        result["selectivity_pct"] = 100 * np.clip(ratio, *self.settings["evaluation"]["ratio_clip"])
        for j, name in enumerate(self.config.states):
            result[f"carbon_{name}_fraction"] = states[:, j]
        for j, name in enumerate(self.config.states[1:-1]):
            result[f"yield_{name}_pct"] = 100 * states[:, j + 1]
        return result

    def trajectory(self, profile: pd.DataFrame, time_min, *, batch_size=256) -> pd.DataFrame:
        if len(profile) != 1:
            raise ValueError("Supply exactly one catalyst and non-time condition profile")
        times = np.asarray(time_min, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("time_min must be a nonempty vector of finite nonnegative times")
        frame = pd.concat([profile] * len(times), ignore_index=True)
        frame["t_min_v9"] = times
        result = self.predict(frame, batch_size=batch_size)
        result.insert(0, "time_min", times)
        return result

    def save(self, path: Path | str) -> None:
        if self.network is None or self.encoder is None:
            raise RuntimeError("No fitted model to save")
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format_version=1,
            model=self.config.name,
            split=self.split,
            seed=self.seed,
            settings=self.settings,
            encoder=self.encoder.to_dict(),
            context=self.context,
            state_dict={k: v.detach().cpu() for k, v in self.network.state_dict().items()},
            history=self.history,
            metadata=self.metadata,
        )
        with path.open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path: Path | str, *, device="auto") -> "CarbonDynamics":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported model file format")
        result = cls(
            payload["model"], payload["split"], payload["seed"], device, config=payload["settings"]
        )
        result.encoder = FrozenEncoder(**payload["encoder"])
        result.context = payload["context"]
        parameters = {name: payload["state_dict"]["_params." + name] for name in PARAMETER_NAMES}
        result.network = CarbonTransferNetwork(result.config, parameters, result.settings).to(
            result.device
        )
        result.network.load_state_dict(payload["state_dict"], strict=True)
        result.history = payload["history"]
        result.metadata = payload["metadata"]
        return result


def predict_ensemble(models: list[CarbonDynamics], frame: pd.DataFrame) -> pd.DataFrame:
    """Average states/outputs first, then calculate selectivity."""
    if not models:
        raise ValueError("At least one model is required")
    if any(model.config != models[0].config or model.split != models[0].split for model in models):
        raise ValueError("Ensemble members must share model configuration and split")
    frames = [model.predict(frame) for model in models]
    columns = [column for column in frames[0] if column != "row_index"]
    result = frames[0].copy()
    result[columns] = np.mean([values[columns].to_numpy() for values in frames], axis=0)
    cfg = models[0].settings["evaluation"]
    result["selectivity_pct"] = 100 * np.clip(
        result.c3_yield_pct
        / np.maximum(result.conversion_pct, 100 * cfg["ratio_denominator_floor"]),
        *cfg["ratio_clip"],
    )
    return result
