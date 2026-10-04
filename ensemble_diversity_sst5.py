
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel

from groupdro_ensemble import solve_group_dro_ensemble
from srm_ensemble import solve_srm_ensemble
from erm import solve_erm_cross_entropy
from irm import solve_irm_ensemble
from rse import solve_robust_offline_with_proximal, calculate_batch_moments
from kl_rs import solve_kl_rs_ensemble
from helpers import (
    load_predictions_from_files_sst5 as load_predictions_from_files,
    split_train_test_pool,
    evaluate_on_shifted_data,
    evaluate_majority_vote,
    get_alphas_for_equidistant_kl,
    generate_shift_distributions,
    create_label_shifted_dataset,
    calculate_kl_divergence,
)


BASE_TRAINING_FILES = [
    "sst5/sst5_results_train_qwen3_4b.json",
    "sst5/sst5_results_train_phi_4_15b.json",
    "sst5/sst5_results_train_phi3.5_mini_it.json",
    "sst5/sst5_results_train_gpt_oss_20b.json",
    "sst5/sst5_results_train_gemma3_12b.json",
    "sst5/sst5_results_train_mistral_7b_it.json",
    "sst5/sst5_results_train_falcon3_7b_it.json",
    "sst5/sst5_results_train_mistral_small_24b.json",
]


OPTIONAL_EXTRA_FILES = [
    "sst5/sst5_results_train_llama_3.1_8b.json",
    "sst5/sst5_results_train_yi1.5_9b.json",
]

SST5_LABEL_ORDER = ["very negative", "negative", "neutral", "positive", "very positive"]
SST5_GT_KEY = "ground_truth_label"
SHIFT_TYPES = ["minority_target"]


# Real weak-model candidates and alias resolution from weak_sst5.py.
DATASET_DIR = Path("sst5")
FILE_PREFIX = "sst5_results_train_"
COMMON_EXTRA_CANDIDATES = [
    ("falcon3_10b", ["falcon3_10b_it", "falcon3_10b"]),
    ("llama3.1_8b", ["llama3.1_8b_it", "llama_3.1_8b", "llama3_1_8b", "llama_3_1_8b_it"]),
    ("phi3_mini", ["phi3_mini_it", "phi3_mini"]),
    ("qwen3_8b", ["qwen3_8b"]),
    ("qwen3_14b", ["qwen3_14b"]),
]



@dataclass(frozen=True)
class Scenario:
    family: str
    pool_id: str
    indices: Tuple[int, ...]
    stress: Optional[str] = None


def stable_int(text: str) -> int:
    """Stable integer seed component independent of Python's hash randomization."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def short_model_name(path: str) -> str:
    return (
        os.path.basename(path)
        .replace("sst5_results_train_", "")
        .replace(".json", "")
    )


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _strip_dataset_prefix(path: Path) -> str:
    name = path.name
    if name.startswith(FILE_PREFIX):
        name = name[len(FILE_PREFIX):]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def short_name(path: str | Path) -> str:
    return _strip_dataset_prefix(Path(path))


def resolve_one_model(aliases, dataset_dir: Path = DATASET_DIR) -> str:
    """Resolve one model JSON by alias. Raises if not found."""
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.json"))
    alias_norms = [_norm(a) for a in aliases]
    matches = []

    for p in files:
        stem = _strip_dataset_prefix(p)
        stem_norm = _norm(stem)
        for a in alias_norms:
            if stem_norm == a or a in stem_norm or stem_norm in a:
                matches.append(p)
                break

    # Remove duplicates while preserving order.
    uniq = []
    seen = set()
    for p in matches:
        rp = str(p)
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)

    if not uniq:
        raise FileNotFoundError(
            f"Could not resolve model aliases {aliases} in {dataset_dir}. "
            f"Available files: {[p.name for p in files]}"
        )

    # Prefer the shortest filename among matches to avoid accidental broad matches.
    uniq = sorted(uniq, key=lambda p: (len(p.name), p.name))
    return str(uniq[0])


def resolve_optional_model(aliases, dataset_dir: Path = DATASET_DIR) -> str | None:
    try:
        return resolve_one_model(aliases, dataset_dir)
    except FileNotFoundError:
        return None


def resolve_candidate_files(original_files: list[str]) -> list[dict]:
    original_set = {str(Path(p)) for p in original_files}
    candidates = []
    for canonical_name, aliases in COMMON_EXTRA_CANDIDATES:
        p = resolve_optional_model(aliases)
        if p is None:
            print(f"[WARN] Missing candidate {canonical_name}; aliases={aliases}")
            continue
        if str(Path(p)) in original_set:
            print(f"[SKIP] Candidate {canonical_name} resolves to an original-pool file: {p}")
            continue
        candidates.append({"canonical_name": canonical_name, "file": p, "resolved_name": short_name(p)})

    # Deduplicate if aliases accidentally resolve to the same file.
    deduped = []
    seen = set()
    for c in candidates:
        if c["file"] not in seen:
            deduped.append(c)
            seen.add(c["file"])
    return deduped


def estimate_candidate_strengths(
    original_files: list[str],
    candidate_records: list[dict],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Rank candidate real extra models by held-out validation accuracy only."""
    all_files = original_files + [c["file"] for c in candidate_records]
    rows = []

    print("\n=== Ranking real extra candidates by validation accuracy ===")
    for exp_num in range(args.n_experiments):
        seed = args.base_seed + exp_num * 50
        all_preds, labels, M, K, N_total = load_predictions_from_files(all_files, SST5_LABEL_ORDER, gt_key=SST5_GT_KEY)

        train_pool_preds, train_pool_labels, _, _ = split_train_test_pool(
            all_preds, labels, K, args.test_pool_fraction, seed=seed
        )
        _, _, val_preds, val_labels = split_train_test_pool(
            train_pool_preds, train_pool_labels, K, args.validation_fraction, seed=seed
        )

        val_accs = individual_accuracies(val_preds, val_labels)
        for j, c in enumerate(candidate_records):
            idx = len(original_files) + j
            rows.append({
                "experiment": exp_num,
                "candidate_model": c["canonical_name"],
                "resolved_name": c["resolved_name"],
                "file": c["file"],
                "val_acc": float(val_accs[idx]),
            })

    df = pd.DataFrame(rows)
    return df


def select_global_weak_models(candidate_strength_df: pd.DataFrame, n_weak_to_add: int) -> pd.DataFrame:
    avg = (
        candidate_strength_df
        .groupby(["candidate_model", "resolved_name", "file"], as_index=False)
        .agg(mean_val_acc=("val_acc", "mean"), std_val_acc=("val_acc", "std"), n_splits=("val_acc", "count"))
        .sort_values("mean_val_acc", ascending=True)
    )
    return avg.head(n_weak_to_add).reset_index(drop=True)


def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum(axis=-1, keepdims=True)


def temperature_transform(p: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    """
    Temperature-scale a probability vector without changing its argmax.
    T < 1 makes predictions over-confident; T > 1 makes them under-confident.
    """
    p = normalize_probs(p, eps=eps)
    logits = np.log(np.clip(p, eps, 1.0)) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=-1, keepdims=True)


def jitter_probs(p: np.ndarray, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    """Create a near-duplicate predictor by adding tiny log-probability noise."""
    p = normalize_probs(p)
    logits = np.log(np.clip(p, 1e-12, 1.0))
    logits = logits + rng.normal(0.0, sigma, size=logits.shape)
    logits = logits - logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=-1, keepdims=True)


def shuffled_weak_copy(
    p: np.ndarray,
    rng: np.random.Generator,
    uniform_mix: float = 0.50,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Synthetic weak predictor: keep realistic probability shapes from a real model,
    but randomly permute examples so outputs are mostly decoupled from inputs.
    No labels are used.
    """
    idx = rng.permutation(p.shape[0])
    q = p[idx].copy()
    q = temperature_transform(q, temperature)
    uniform = np.full_like(q, 1.0 / q.shape[-1])
    q = (1.0 - uniform_mix) * q + uniform_mix * uniform
    return normalize_probs(q)


def individual_accuracies(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Hard-label accuracy for each model in a prediction tensor [M, N, K]."""
    return np.array([
        np.mean(np.argmax(preds[m], axis=1) == labels) if len(labels) else np.nan
        for m in range(preds.shape[0])
    ])


def hardest_class_from_validation(val_preds: np.ndarray, val_labels: np.ndarray, K: int) -> int:
    """Replicates your hardest-class heuristic, computed for the current pool."""
    M = val_preds.shape[0]
    class_acc = np.full((M, K), np.nan)
    for m in range(M):
        pred_m = np.argmax(val_preds[m], axis=1)
        for k in range(K):
            idx = np.where(val_labels == k)[0]
            if len(idx) > 0:
                class_acc[m, k] = np.mean(pred_m[idx] == val_labels[idx])
    with np.errstate(all="ignore"):
        mean_by_class = np.nanmean(class_acc, axis=0)
    mean_by_class = np.where(np.isfinite(mean_by_class), mean_by_class, 1.0)
    return int(np.argmin(mean_by_class))


def avg_pairwise_prediction_corr(preds: np.ndarray) -> Tuple[float, float]:
    """Average and maximum off-diagonal correlation of flattened probability outputs."""
    M = preds.shape[0]
    if M < 2:
        return float("nan"), float("nan")
    X = preds.reshape(M, -1)
    # Avoid NaNs from constant predictors.
    X = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(denom, 1e-12, None)
    corr = Xn @ Xn.T
    off = corr[np.triu_indices(M, k=1)]
    off = off[np.isfinite(off)]
    if len(off) == 0:
        return float("nan"), float("nan")
    return float(np.mean(off)), float(np.max(off))


def pool_diagnostics(val_preds: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
    accs = individual_accuracies(val_preds, val_labels)
    avg_corr, max_corr = avg_pairwise_prediction_corr(val_preds)
    return {
        "pool_val_acc_mean": float(np.nanmean(accs)),
        "pool_val_acc_min": float(np.nanmin(accs)),
        "pool_val_acc_max": float(np.nanmax(accs)),
        "pool_avg_pair_corr": avg_corr,
        "pool_max_pair_corr": max_corr,
    }


def build_scenarios(
    m_original: int,
    n_extra_loaded: int,
    model_names: Sequence[str],
    subsets_per_size: int,
    scenario_seed: int,
    include_random_subsets: bool = True,
    include_stress: bool = True,
    real_weak_indices: Sequence[int] = (),
) -> List[Scenario]:
    rng = np.random.default_rng(scenario_seed)
    base = tuple(range(m_original))
    scenarios: List[Scenario] = [Scenario("original", "original_8", base)]

    # Leave-one-out: directly answers "when models are removed".
    for drop_idx in range(m_original):
        keep = tuple(i for i in base if i != drop_idx)
        scenarios.append(
            Scenario(
                "leave_one_out",
                f"drop_{drop_idx}_{model_names[drop_idx]}",
                keep,
            )
        )

    # Random subsets by cardinality: sensitivity to number and composition.
    if include_random_subsets:
        for size in range(2, m_original):
            combos = list(itertools.combinations(base, size))
            rng.shuffle(combos)
            for j, combo in enumerate(combos[: min(subsets_per_size, len(combos))]):
                scenarios.append(
                    Scenario("random_subset", f"size_{size}_sample_{j}", tuple(combo))
                )

    # Full-pool stress tests: directly answer reviewer concerns.
    if include_stress:
        scenarios.extend(
            [
                Scenario("plus_weak", f"original_8_plus_real_weak_x{len(real_weak_indices)}", base + tuple(real_weak_indices)),
                Scenario("plus_correlated", "original_8_plus_corr_x4", base, stress="correlated"),
                Scenario("plus_uncalibrated", "original_8_plus_uncal_x4", base, stress="uncalibrated"),
                Scenario("plus_weak_uncalibrated", "original_8_plus_weak_uncal_x4", base, stress="weak_uncalibrated"),
            ]
        )

    # Actual extra models, when available, are more convincing than synthetic weak models.
    if n_extra_loaded > 0:
        all_loaded = tuple(range(m_original + n_extra_loaded))
        scenarios.append(
            Scenario(
                "plus_actual_extra",
                f"original_8_plus_{n_extra_loaded}_actual_extra",
                all_loaded,
            )
        )

    return scenarios


def append_stress_models(
    train_preds: np.ndarray,
    val_preds: np.ndarray,
    test_preds: np.ndarray,
    val_labels: np.ndarray,
    scenario: Scenario,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Append the remaining synthetic stress predictors to the selected pool."""
    stress = scenario.stress
    if stress is None:
        return train_preds, val_preds, test_preds, []

    M = train_preds.shape[0]
    n_extra = args.stress_extra_models
    val_accs = individual_accuracies(val_preds, val_labels)
    best_idx = int(np.nanargmax(val_accs)) if len(val_accs) else 0
    order = list(np.argsort(-np.nan_to_num(val_accs, nan=-1.0)))
    if not order:
        order = list(range(M))

    extra_train, extra_val, extra_test, extra_names = [], [], [], []

    if stress == "correlated":
        # Near-duplicates of the validation-best model; avg pairwise correlations
        # in the diagnostics should be close to one.
        for i in range(n_extra):
            extra_train.append(jitter_probs(train_preds[best_idx], rng, sigma=args.corr_noise))
            extra_val.append(jitter_probs(val_preds[best_idx], rng, sigma=args.corr_noise))
            extra_test.append(jitter_probs(test_preds[best_idx], rng, sigma=args.corr_noise))
            extra_names.append(f"corr_copy_best_{i}")

    elif stress == "uncalibrated":
        temps = [0.15, 0.25, 4.0, 8.0]
        for i in range(n_extra):
            base_idx = order[i % M]
            T = temps[i % len(temps)]
            extra_train.append(temperature_transform(train_preds[base_idx], T))
            extra_val.append(temperature_transform(val_preds[base_idx], T))
            extra_test.append(temperature_transform(test_preds[base_idx], T))
            extra_names.append(f"temp_scaled_T{T:g}_{i}")

    elif stress == "weak_uncalibrated":
        # Very harsh condition: shuffled/noisy predictors that are also made
        # over-confident, so a probability-weighted ensemble can be misled.
        for i in range(n_extra):
            base_idx = order[i % M]
            extra_train.append(
                shuffled_weak_copy(
                    train_preds[base_idx], rng,
                    uniform_mix=args.weak_uncal_uniform_mix,
                    temperature=args.weak_uncal_temperature,
                )
            )
            extra_val.append(
                shuffled_weak_copy(
                    val_preds[base_idx], rng,
                    uniform_mix=args.weak_uncal_uniform_mix,
                    temperature=args.weak_uncal_temperature,
                )
            )
            extra_test.append(
                shuffled_weak_copy(
                    test_preds[base_idx], rng,
                    uniform_mix=args.weak_uncal_uniform_mix,
                    temperature=args.weak_uncal_temperature,
                )
            )
            extra_names.append(f"weak_overconf_{i}")

    else:
        raise ValueError(f"Unknown stress type: {stress}")

    train_out = np.concatenate([train_preds, np.stack(extra_train, axis=0)], axis=0)
    val_out = np.concatenate([val_preds, np.stack(extra_val, axis=0)], axis=0)
    test_out = np.concatenate([test_preds, np.stack(extra_test, axis=0)], axis=0)
    return train_out, val_out, test_out, extra_names


def train_ensemble_weights(
    train_preds: np.ndarray,
    train_labels: np.ndarray,
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    q_source: np.ndarray,
    K: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, float], Dict[str, str]]:
    """Train selected methods and return method -> weight vector."""
    M = train_preds.shape[0]
    weights: Dict[str, Optional[np.ndarray]] = {}
    times: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    def run(name: str, fn):
        start = time.perf_counter()
        try:
            weights[name] = fn()
            times[name] = time.perf_counter() - start
        except Exception as exc:  # keep large sweeps running
            weights[name] = None
            times[name] = time.perf_counter() - start
            errors[name] = repr(exc)
            print(f"[WARN] {name} failed for M={M}: {exc}")

    selected = set(args.methods)

    if "ERM" in selected:
        run("ERM", lambda: solve_erm_cross_entropy(train_preds, train_labels, M))

    if "RSE" in selected:
        def _rse():
            A_matrices, b_vectors = calculate_batch_moments(train_preds, train_labels, M, K)
            _, w_robust, _, _ = solve_robust_offline_with_proximal(
                A_matrices,
                b_vectors,
                q_source,
                M,
                K,
                args.tau_target_fraction,
                T_max_iter=args.rse_max_iter,
                epsilon_convergence=args.rse_epsilon,
                eta=args.rse_eta,
            )
            return w_robust
        run("RSE", _rse)

    if "DRO" in selected:
        run("DRO", lambda: solve_group_dro_ensemble(train_preds, train_labels, M, K)[0])

    if "SRM" in selected:
        run("SRM", lambda: solve_srm_ensemble(train_preds, train_labels, M, K)[0])

    if "KL-RS" in selected:
        run(
            "KL-RS",
            lambda: solve_kl_rs_ensemble(
                train_preds,
                train_labels,
                M,
                tau_factor=args.klrs_tau_factor,
                lambda_min=args.klrs_lambda_min,
                lambda_max=args.klrs_lambda_max,
            )[0],
        )

    if "IRM" in selected:
        run(
            "IRM",
            lambda: solve_irm_ensemble(
                train_preds,
                train_labels,
                q_source,
                M,
                lambda_irm=args.irm_lambda,
                lr=args.irm_lr,
            )[0],
        )

    if "Uniform" in selected:
        weights["Uniform"] = np.full(M, 1.0 / M)
        times["Uniform"] = 0.0

    if "BestValIndividual" in selected:
        val_accs = individual_accuracies(val_preds, val_labels)
        best = int(np.nanargmax(val_accs))
        w = np.zeros(M)
        w[best] = 1.0
        weights["BestValIndividual"] = w
        times["BestValIndividual"] = 0.0

    if "MajorityVote" in selected:
        weights["MajorityVote"] = None
        times["MajorityVote"] = 0.0

    return weights, times, errors


def normalize_weight_vector(w: Optional[np.ndarray], M: int) -> np.ndarray:
    if w is None:
        return np.full(M, 1.0 / M)
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != M or not np.all(np.isfinite(w)) or abs(float(np.sum(w))) < 1e-12:
        return np.full(M, 1.0 / M)
    return w / np.sum(w)


def evaluate_scenario(
    scenario: Scenario,
    model_names: Sequence[str],
    train_preds_base: np.ndarray,
    train_labels: np.ndarray,
    val_preds_base: np.ndarray,
    val_labels: np.ndarray,
    test_preds_base: np.ndarray,
    test_labels: np.ndarray,
    K: int,
    exp_num: int,
    exp_seed: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    idx = np.array(scenario.indices, dtype=int)
    train_preds = train_preds_base[idx].copy()
    val_preds = val_preds_base[idx].copy()
    test_preds = test_preds_base[idx].copy()
    names = [model_names[i] for i in idx]

    rng = np.random.default_rng(exp_seed + stable_int(scenario.pool_id))
    train_preds, val_preds, test_preds, extra_names = append_stress_models(
        train_preds, val_preds, test_preds, val_labels, scenario, args, rng
    )
    names = names + extra_names

    M = train_preds.shape[0]
    q_source = np.bincount(train_labels, minlength=K) / len(train_labels)
    q_test_pool = np.bincount(test_labels, minlength=K) / len(test_labels)
    hardest_class = hardest_class_from_validation(val_preds, val_labels, K)

    magnitudes = get_alphas_for_equidistant_kl(
        q_source=q_source,
        shift_type="minority_target",
        K=K,
        target_class_index=hardest_class,
        num_points=args.kl_grid_points,
    )

    weights, train_times, train_errors = train_ensemble_weights(
        train_preds, train_labels, val_preds, val_labels, q_source, K, args
    )

    diag = pool_diagnostics(val_preds, val_labels)
    meta = {
        "scenario_family": scenario.family,
        "pool_id": scenario.pool_id,
        "stress": scenario.stress or "none",
        "pool_size": M,
        "base_indices": list(map(int, scenario.indices)),
        "model_names": names,
        "hardest_class_index": int(hardest_class),
        "hardest_class_name": SST5_LABEL_ORDER[hardest_class] if hardest_class < len(SST5_LABEL_ORDER) else str(hardest_class),
        "train_times": train_times,
        "train_errors": train_errors,
        **diag,
    }

    rows: List[Dict[str, object]] = []
    test_count = 0
    for shift_type in SHIFT_TYPES:
        for magnitude in magnitudes:
            target_dist = generate_shift_distributions(
                q_test_pool, shift_type, magnitude, K, hardest_class
            )
            for trial in range(args.n_trials_per_shift):
                test_count += 1
                shifted_preds, shifted_labels = create_label_shifted_dataset(
                    test_preds,
                    test_labels,
                    target_dist,
                    args.generated_test_set_size,
                    seed=exp_seed + 100000 * stable_int(scenario.pool_id) % 1000003 + test_count,
                )
                if shifted_preds is None:
                    continue

                actual_dist = np.bincount(shifted_labels, minlength=K) / len(shifted_labels)
                kl_div = calculate_kl_divergence(actual_dist, q_source)

                for method_name, w_vec in weights.items():
                    if method_name in train_errors:
                        continue
                    if method_name == "MajorityVote":
                        eval_result = evaluate_majority_vote(shifted_preds, shifted_labels)
                    else:
                        w = normalize_weight_vector(w_vec, M)
                        eval_result = evaluate_on_shifted_data(w, shifted_preds, shifted_labels)

                    rows.append(
                        {
                            "exp": exp_num,
                            "exp_seed": exp_seed,
                            "scenario_family": scenario.family,
                            "pool_id": scenario.pool_id,
                            "stress": scenario.stress or "none",
                            "pool_size": M,
                            "pool_models": ";".join(names),
                            "method": method_name,
                            "shift_type": shift_type,
                            "magnitude": float(magnitude),
                            "trial": trial,
                            "kl_divergence": float(kl_div),
                            "accuracy": float(eval_result["accuracy"]),
                            "hardest_class_index": int(hardest_class),
                            "hardest_class_name": meta["hardest_class_name"],
                            "pool_val_acc_mean": diag["pool_val_acc_mean"],
                            "pool_val_acc_min": diag["pool_val_acc_min"],
                            "pool_val_acc_max": diag["pool_val_acc_max"],
                            "pool_avg_pair_corr": diag["pool_avg_pair_corr"],
                            "pool_max_pair_corr": diag["pool_max_pair_corr"],
                            "train_time_sec": float(train_times.get(method_name, np.nan)),
                        }
                    )

    return rows, meta


def sem(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def summarize_and_plot(df: pd.DataFrame, out_dir: str, methods: Sequence[str]) -> None:
    if df.empty:
        print("[WARN] No rows to summarize.")
        return

    # Table 1: scenario-family summary.
    summary = (
        df.groupby(["scenario_family", "method"], dropna=False)
        .agg(
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            sem_acc=("accuracy", sem),
            q10_acc=("accuracy", lambda x: float(np.quantile(x, 0.10))),
            worst_acc=("accuracy", "min"),
            mean_pool_size=("pool_size", "mean"),
            mean_pool_corr=("pool_avg_pair_corr", "mean"),
            mean_val_acc_min=("pool_val_acc_min", "mean"),
            n=("accuracy", "size"),
        )
        .reset_index()
    )
    for col in ["mean_acc", "std_acc", "sem_acc", "q10_acc", "worst_acc"]:
        summary[col + "_pct"] = 100.0 * summary[col]
    summary.to_csv(os.path.join(out_dir, "summary_by_scenario_method.csv"), index=False)

    # Table 2: per-pool summary. Useful for leave-one-out rows.
    pool_summary = (
        df.groupby(["scenario_family", "pool_id", "pool_size", "method"], dropna=False)
        .agg(
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            sem_acc=("accuracy", sem),
            q10_acc=("accuracy", lambda x: float(np.quantile(x, 0.10))),
            worst_acc=("accuracy", "min"),
            mean_pool_corr=("pool_avg_pair_corr", "mean"),
            mean_val_acc_min=("pool_val_acc_min", "mean"),
            n=("accuracy", "size"),
        )
        .reset_index()
    )
    for col in ["mean_acc", "std_acc", "sem_acc", "q10_acc", "worst_acc"]:
        pool_summary[col + "_pct"] = 100.0 * pool_summary[col]
    pool_summary.to_csv(os.path.join(out_dir, "summary_by_pool_method.csv"), index=False)

    # Paired RSE margins by exact evaluation unit.
    unit_cols = ["scenario_family", "pool_id", "exp", "shift_type", "magnitude", "trial"]
    wide = df.pivot_table(index=unit_cols, columns="method", values="accuracy", aggfunc="mean").reset_index()
    if "RSE" in wide.columns:
        baseline_cols = [
            c for c in wide.columns
            if c not in unit_cols and c != "RSE" and pd.api.types.is_numeric_dtype(wide[c])
        ]
        if baseline_cols:
            wide["best_baseline_acc"] = wide[baseline_cols].max(axis=1)
            wide["rse_minus_best_baseline"] = wide["RSE"] - wide["best_baseline_acc"]
            wide["rse_beats_best_baseline"] = wide["rse_minus_best_baseline"] >= 0

            margin_summary_rows = []
            for fam, g in wide.groupby("scenario_family"):
                diffs = g["rse_minus_best_baseline"].dropna().to_numpy()
                if len(diffs) == 0:
                    continue
                t_res = ttest_1samp(diffs, popmean=0.0, nan_policy="omit") if len(diffs) > 1 else None
                margin_summary_rows.append(
                    {
                        "scenario_family": fam,
                        "mean_margin_pct": 100.0 * float(np.mean(diffs)),
                        "sem_margin_pct": 100.0 * sem(diffs),
                        "ci95_margin_pct": 100.0 * 1.96 * sem(diffs),
                        "win_rate_vs_best_baseline": float(np.mean(diffs >= 0.0)),
                        "n_units": int(len(diffs)),
                        "p_value_ttest_margin_gt_0_two_sided": float(t_res.pvalue) if t_res is not None else float("nan"),
                    }
                )
            margin_summary = pd.DataFrame(margin_summary_rows)
            margin_summary.to_csv(
                os.path.join(out_dir, "rse_vs_best_baseline_by_scenario.csv"), index=False
            )

            # RSE vs each baseline, paired.
            each_rows = []
            for fam, g in wide.groupby("scenario_family"):
                for b in baseline_cols:
                    valid = g[["RSE", b]].dropna()
                    if len(valid) == 0:
                        continue
                    diffs = (valid["RSE"] - valid[b]).to_numpy()
                    if len(diffs) > 1 and not np.allclose(diffs, diffs[0]):
                        _, p_val = ttest_rel(valid["RSE"], valid[b], nan_policy="omit")
                    else:
                        p_val = 1.0
                    each_rows.append(
                        {
                            "scenario_family": fam,
                            "baseline": b,
                            "mean_rse_minus_baseline_pct": 100.0 * float(np.mean(diffs)),
                            "sem_pct": 100.0 * sem(diffs),
                            "ci95_pct": 100.0 * 1.96 * sem(diffs),
                            "win_rate": float(np.mean(diffs >= 0.0)),
                            "n_units": int(len(diffs)),
                            "paired_ttest_p_value": float(p_val),
                        }
                    )
            pd.DataFrame(each_rows).to_csv(
                os.path.join(out_dir, "rse_vs_each_baseline_by_scenario.csv"), index=False
            )

            plot_pool_size_accuracy(df, out_dir, methods)
            plot_stress_margins(margin_summary, out_dir)
            write_rebuttal_snippet(summary, margin_summary, out_dir)

    print("\n=== Scenario-method summary, mean accuracy (%) ===")
    printable = summary.copy()
    printable["mean ± sem"] = printable.apply(
        lambda r: f"{r['mean_acc_pct']:.2f} ± {r['sem_acc_pct']:.2f}", axis=1
    )
    table = printable.pivot_table(
        index="scenario_family", columns="method", values="mean_acc_pct", aggfunc="mean"
    )
    print(table.round(2).to_string())


def plot_pool_size_accuracy(df: pd.DataFrame, out_dir: str, methods: Sequence[str]) -> None:
    subset_fams = {"original", "leave_one_out", "random_subset"}
    d = df[df["scenario_family"].isin(subset_fams)].copy()
    if d.empty:
        return
    # Average within each pool/experiment first so large shift grids do not
    # change the interpretation of the uncertainty bars.
    unit = (
        d.groupby(["pool_size", "pool_id", "exp", "method"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
    )
    g = (
        unit.groupby(["pool_size", "method"], dropna=False)
        .agg(mean_acc=("accuracy", "mean"), sem_acc=("accuracy", sem))
        .reset_index()
    )

    preferred = [m for m in ["RSE", "ERM", "DRO", "KL-RS", "SRM", "IRM", "Uniform", "BestValIndividual"] if m in methods]
    plt.figure(figsize=(6.0, 3.8))
    for method in preferred:
        gd = g[g["method"] == method].sort_values("pool_size")
        if gd.empty:
            continue
        plt.errorbar(
            gd["pool_size"],
            100.0 * gd["mean_acc"],
            yerr=100.0 * 1.96 * gd["sem_acc"],
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Number of available base models")
    plt.ylabel("Accuracy (%)")
    plt.title("Sensitivity to model-pool size and composition")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pool_size_accuracy.pdf"), bbox_inches="tight")
    plt.close()


def plot_stress_margins(margin_summary: pd.DataFrame, out_dir: str) -> None:
    if margin_summary.empty:
        return
    order = [
        "original",
        "leave_one_out",
        "random_subset",
        "plus_weak",
        "plus_correlated",
        "plus_uncalibrated",
        "plus_weak_uncalibrated",
        "plus_actual_extra",
    ]
    d = margin_summary.set_index("scenario_family").reindex(order).dropna(how="all").reset_index()
    if d.empty:
        return
    plt.figure(figsize=(7.2, 3.8))
    x = np.arange(len(d))
    plt.bar(x, d["mean_margin_pct"], yerr=d["ci95_margin_pct"], capsize=3)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, d["scenario_family"], rotation=25, ha="right")
    plt.ylabel("RSE - best baseline (pp)")
    plt.title("RSE margin under pool-composition stress tests")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stress_rse_margin.pdf"), bbox_inches="tight")
    plt.close()


def write_rebuttal_snippet(summary: pd.DataFrame, margin_summary: pd.DataFrame, out_dir: str) -> None:
    lines = []
    lines.append("Model-pool sensitivity rebuttal snippet")
    lines.append("=======================================")
    lines.append("")
    lines.append(
        "We added a model-pool sensitivity study using the same data splits, label-shift protocol, "
        "and baselines as the main experiment. Starting from the original eight-model pool, we evaluated "
        "(i) all leave-one-out pools, (ii) random subsets of sizes 2--7, and (iii) stress pools obtained by "
        "adding real weak predictors selected by validation accuracy, near-duplicate highly correlated predictors, temperature-scaled "
        "uncalibrated predictors, and weak over-confident predictors."
    )
    lines.append("")

    if not margin_summary.empty:
        lines.append("RSE margin over the strongest baseline on the same evaluation units:")
        for _, r in margin_summary.sort_values("scenario_family").iterrows():
            lines.append(
                f"  - {r['scenario_family']}: {r['mean_margin_pct']:.2f} ± "
                f"{r['ci95_margin_pct']:.2f} pp, win-rate={100*r['win_rate_vs_best_baseline']:.1f}%, "
                f"n={int(r['n_units'])}"
            )
        lines.append("")

    if not summary.empty and "RSE" in set(summary["method"]):
        rse = summary[summary["method"] == "RSE"].sort_values("scenario_family")
        lines.append("RSE mean accuracies by scenario:")
        for _, r in rse.iterrows():
            lines.append(
                f"  - {r['scenario_family']}: {r['mean_acc_pct']:.2f} ± {r['sem_acc_pct']:.2f}% "
                f"(q10={r['q10_acc_pct']:.2f}%, worst={r['worst_acc_pct']:.2f}%)"
            )
        lines.append("")

    lines.append(
        "Replace the numbers above into the rebuttal. A conservative phrasing is: "
        "'These experiments indicate that the gains are not driven by a single selected collection: "
        "RSE remains competitive across leave-one-out and random subsets, and its margin does not collapse "
        "when weak, correlated, or miscalibrated predictors are added.'"
    )

    with open(os.path.join(out_dir, "rebuttal_snippet.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSE model-pool sensitivity rebuttal experiment")

    parser.add_argument("--out_dir", type=str, default="rebuttal_model_pool_sensitivity")
    parser.add_argument("--n_experiments", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--scenario_seed", type=int, default=2026)
    parser.add_argument("--test_pool_fraction", type=float, default=0.10)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--generated_test_set_size", type=int, default=1000)
    parser.add_argument("--n_trials_per_shift", type=int, default=5)
    parser.add_argument("--kl_grid_points", type=int, default=11)
    parser.add_argument("--subsets_per_size", type=int, default=8)
    parser.add_argument("--stress_extra_models", type=int, default=4)
    parser.add_argument("--n_weak_to_add", type=int, default=3,
                        help="Number of real extra models with the lowest mean validation accuracy to add.")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Write raw_results_checkpoint.csv every N scenarios.")

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RSE", "ERM", "DRO", "SRM", "KL-RS", "IRM", "Uniform", "BestValIndividual"],
        choices=["RSE", "ERM", "DRO", "SRM", "KL-RS", "IRM", "Uniform", "BestValIndividual", "MajorityVote"],
    )

    parser.add_argument("--skip_random_subsets", action="store_true")
    parser.add_argument("--skip_stress", action="store_true")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Optional smoke-test limit.")

    parser.add_argument(
        "--auto_extra",
        action="store_true",
        help="Include OPTIONAL_EXTRA_FILES when they exist on disk.",
    )
    parser.add_argument(
        "--extra_model_files",
        nargs="*",
        default=[],
        help="Additional real model prediction files to append as plus_actual_extra.",
    )

    # Solver hyperparameters copied from your original script.
    parser.add_argument("--tau_target_fraction", type=float, default=0.70)
    parser.add_argument("--rse_max_iter", type=int, default=2000)
    parser.add_argument("--rse_epsilon", type=float, default=1e-5)
    parser.add_argument("--rse_eta", type=float, default=0.2)
    parser.add_argument("--klrs_tau_factor", type=float, default=1.1)
    parser.add_argument("--klrs_lambda_min", type=float, default=1e-5)
    parser.add_argument("--klrs_lambda_max", type=float, default=1000.0)
    parser.add_argument("--irm_lambda", type=float, default=0.4)
    parser.add_argument("--irm_lr", type=float, default=0.01)

    # Stress-test parameters.
    parser.add_argument("--weak_uniform_mix", type=float, default=0.50)
    parser.add_argument("--weak_uncal_uniform_mix", type=float, default=0.02)
    parser.add_argument("--weak_uncal_temperature", type=float, default=0.15)
    parser.add_argument("--corr_noise", type=float, default=0.005)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    extra_files: List[str] = []
    if args.auto_extra:
        extra_files.extend([p for p in OPTIONAL_EXTRA_FILES if os.path.exists(p)])
    for p in args.extra_model_files:
        if os.path.exists(p):
            extra_files.append(p)
        else:
            print(f"[WARN] Extra file does not exist and will be skipped: {p}")

    # Remove duplicates while preserving order.
    seen = set()
    extra_files = [p for p in extra_files if not (p in seen or seen.add(p))]

    training_files = BASE_TRAINING_FILES + extra_files
    real_weak_indices: List[int] = []
    if not args.skip_stress:
        if args.n_weak_to_add < 1:
            raise ValueError("--n_weak_to_add must be at least 1.")
        candidate_records = resolve_candidate_files(BASE_TRAINING_FILES)
        if not candidate_records:
            raise RuntimeError("No common extra candidate files were found. Check filenames and DATASET_DIR.")

        strength_df = estimate_candidate_strengths(BASE_TRAINING_FILES, candidate_records, args)
        strength_df.to_csv(
            os.path.join(args.out_dir, "weak_candidate_validation_accuracies_by_split.csv"), index=False
        )
        selected = select_global_weak_models(strength_df, args.n_weak_to_add)
        selected.to_csv(os.path.join(args.out_dir, "selected_real_weak_models.csv"), index=False)

        print("\n=== Selected real weak models to add ===")
        print((selected.assign(mean_val_acc_pct=100 * selected["mean_val_acc"])
               [["candidate_model", "resolved_name", "mean_val_acc_pct", "file"]]
               .round({"mean_val_acc_pct": 2})
               .to_string(index=False)))

        # Reuse an already-loaded optional extra; do not add a duplicate predictor.
        loaded_paths = {Path(p).resolve(): i for i, p in enumerate(training_files)}
        for p in selected["file"]:
            resolved_path = Path(p).resolve()
            if resolved_path not in loaded_paths:
                loaded_paths[resolved_path] = len(training_files)
                training_files.append(p)
            real_weak_indices.append(loaded_paths[resolved_path])
    model_names = [short_model_name(p) for p in training_files]

    print("Loading prediction files:")
    for i, p in enumerate(training_files):
        tag = "base" if i < len(BASE_TRAINING_FILES) else "extra"
        print(f"  [{i:02d}] ({tag}) {p}")

    all_predictions, all_labels, M_loaded, K, N_total = load_predictions_from_files(
        training_files, SST5_LABEL_ORDER, gt_key=SST5_GT_KEY
    )
    if M_loaded != len(training_files):
        raise RuntimeError(f"Loader returned M={M_loaded}, but {len(training_files)} files were requested.")

    scenarios = build_scenarios(
        m_original=len(BASE_TRAINING_FILES),
        n_extra_loaded=len(extra_files),
        model_names=model_names,
        subsets_per_size=args.subsets_per_size,
        scenario_seed=args.scenario_seed,
        include_random_subsets=not args.skip_random_subsets,
        include_stress=not args.skip_stress,
        real_weak_indices=real_weak_indices,
    )
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]

    with open(os.path.join(args.out_dir, "scenario_plan.json"), "w") as f:
        json.dump([s.__dict__ for s in scenarios], f, indent=2)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nRunning {len(scenarios)} scenarios × {args.n_experiments} experiment splits")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Outputs will be written to: {args.out_dir}\n")

    all_rows: List[Dict[str, object]] = []
    all_meta: List[Dict[str, object]] = []
    scenario_counter = 0

    for exp_num in range(args.n_experiments):
        current_seed = args.base_seed + exp_num * 50
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {exp_num + 1}/{args.n_experiments}  seed={current_seed}")
        print("=" * 80)

        train_pool_preds, train_pool_labels, test_pool_preds, test_pool_labels = split_train_test_pool(
            all_predictions, all_labels, K, args.test_pool_fraction, seed=current_seed
        )
        actual_train_preds, actual_train_labels, val_preds, val_labels = split_train_test_pool(
            train_pool_preds, train_pool_labels, K, args.validation_fraction, seed=current_seed
        )

        for scenario in scenarios:
            scenario_counter += 1
            print(
                f"[{scenario_counter:04d}] exp={exp_num + 1:02d} "
                f"family={scenario.family:<24} pool={scenario.pool_id}"
            )
            try:
                rows, meta = evaluate_scenario(
                    scenario=scenario,
                    model_names=model_names,
                    train_preds_base=actual_train_preds,
                    train_labels=actual_train_labels,
                    val_preds_base=val_preds,
                    val_labels=val_labels,
                    test_preds_base=test_pool_preds,
                    test_labels=test_pool_labels,
                    K=K,
                    exp_num=exp_num,
                    exp_seed=current_seed,
                    args=args,
                )
                all_rows.extend(rows)
                meta = dict(meta)
                meta["exp"] = exp_num
                meta["exp_seed"] = current_seed
                all_meta.append(meta)
            except Exception as exc:
                print(f"[ERROR] Scenario failed: {scenario}. Error: {exc}")
                all_meta.append(
                    {
                        "exp": exp_num,
                        "exp_seed": current_seed,
                        "scenario_family": scenario.family,
                        "pool_id": scenario.pool_id,
                        "error": repr(exc),
                    }
                )

            if args.checkpoint_every and scenario_counter % args.checkpoint_every == 0:
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(args.out_dir, "raw_results_checkpoint.csv"), index=False
                )
                with open(os.path.join(args.out_dir, "scenario_metadata_checkpoint.json"), "w") as f:
                    json.dump(all_meta, f, indent=2)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "raw_results.csv"), index=False)
    with open(os.path.join(args.out_dir, "scenario_metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    summarize_and_plot(df, args.out_dir, args.methods)
    print(f"\nDone. Main files written under: {args.out_dir}")


if __name__ == "__main__":
    main()