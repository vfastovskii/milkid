#!/usr/bin/env python3
"""
Calculate top-k attention success rates from per-bag conformer CSV files.

RMSD metric
-----------
A bag is RMSD-valid if:
    min(min_best_rmsd_to_exp) <= --rmsd-threshold

For each RMSD-valid bag, top-k RMSD success is defined as:
    at least one conformer among the k highest attention_weight values has
    best_rmsd_to_exp <= --rmsd-threshold

Normalized O3A metric
---------------------
If --norm-o3a-threshold is provided, the script also computes:
    norm_o3a_to_exp = o3a_to_exp / max_03a

A bag is normalized-O3A-valid if:
    max(norm_o3a_to_exp) >= --norm-o3a-threshold

For each normalized-O3A-valid bag, top-k normalized-O3A success is defined as:
    at least one conformer among the k highest attention_weight values has
    norm_o3a_to_exp >= --norm-o3a-threshold

Hypergeometric random baseline
------------------------------
For each eligible bag, the random baseline estimates the probability of retrieving
at least one successful conformer when randomly selecting k conformers without
replacement:

    P(success) = 1 - C(N - H, k) / C(N, k)

where:
    N = number of conformers in the eligible bag
    H = number of successful conformers in the bag
    k = min(requested top-k, N)

The global random baseline is the average of these per-bag probabilities over
eligible bags.

Paired t-test
-------------
For each metric and top-k value, the script performs a paired one-sample
t-test on the per-bag differences:

    difference_i = model_success_i - random_success_probability_i

where model_success_i is 0/1 and random_success_probability_i is the
hypergeometric probability for the same bag.

Barplot
-------
The script can also generate a publication-style grouped barplot comparing
model top-k success rates against the hypergeometric random baseline.
Significant model-vs-random differences are marked above the model bars.

Ranking metrics
---------------
For each valid bag, the script also calculates full-ranking metrics after
sorting conformers by descending attention weight:

    BEDROC(alpha), with default alpha = 80
    nDCG@k, with default k = 10

For RMSD, relevant conformers are those with best_rmsd_to_exp <= threshold.
For normalized O3A, relevant conformers are those with norm_o3a_to_exp >= threshold.
"""

from __future__ import annotations

import argparse
from math import comb, exp, log2
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


UNKNOWN_SERIES = "UNKNOWN_SERIES"
RMSD_METRIC = "rmsd"
NORM_O3A_METRIC = "norm_o3a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate top-k attention success rates for conformer CSV files."
    )

    parser.add_argument("input_dir", type=Path, help="Directory containing CSV files.")

    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        required=True,
        help="RMSD cutoff. Lower or equal is successful.",
    )

    parser.add_argument(
        "--norm-o3a-threshold",
        type=float,
        default=None,
        help=(
            "Optional normalized O3A cutoff. If provided, computes "
            "norm_o3a_to_exp = o3a_to_exp / max_03a. "
            "Greater or equal is successful."
        ),
    )

    parser.add_argument(
        "--bedroc-alpha",
        type=float,
        default=80.0,
        help=(
            "BEDROC alpha parameter. Higher values emphasize earlier ranks more strongly. "
            "Default: 80."
        ),
    )

    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="Cutoff for nDCG@k. Default: 10.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values to evaluate. Default: 1 3 5.",
    )

    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--attention-col", type=str, default="attention_weight")
    parser.add_argument("--rmsd-col", type=str, default="best_rmsd_to_exp")
    parser.add_argument("--min-rmsd-col", type=str, default="min_best_rmsd_to_exp")
    parser.add_argument("--o3a-col", type=str, default="o3a_to_exp")
    parser.add_argument("--max-o3a-col", type=str, default="max_03a")

    parser.add_argument(
        "--bag-id-col",
        type=str,
        default=None,
        help="Optional bag ID column. If omitted, each CSV file is one bag.",
    )

    parser.add_argument(
        "--series-col",
        type=str,
        default="Series",
        help="Series column. Default: Series. Pass empty string to disable.",
    )

    parser.add_argument(
        "--instance-id-col",
        type=str,
        default="instance_id",
        help=(
            "Column containing conformer/instance identifiers for the representatives log. "
            "Default: instance_id. If missing, row indices are used."
        ),
    )

    parser.add_argument(
        "--representatives-top-k",
        type=int,
        nargs="+",
        default=[1, 5],
        help=(
            "Top-attended conformers to list for each representative bag in the "
            "series representatives log. Default: 1 5."
        ),
    )

    parser.add_argument(
        "--make-representatives-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create a series -> representatives log and CSV with top-attended "
            "conformers and RMSD bioactive-like flags. Default: true."
        ),
    )

    parser.add_argument(
        "--descending-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank larger attention weights first. Default: true.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("topk_success"),
        help="Output prefix.",
    )

    parser.add_argument(
        "--ttest-alternative",
        type=str,
        choices=["greater", "two-sided", "less"],
        default="greater",
        help=(
            "Alternative hypothesis for the paired one-sample t-test on "
            "per-bag differences: model_success - random_success_probability. "
            "Default: greater, meaning model > random baseline."
        ),
    )

    parser.add_argument(
        "--ttest-alpha",
        type=float,
        default=0.05,
        help="Significance level used only for reporting. Default: 0.05.",
    )

    parser.add_argument(
        "--make-barplot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a publication-style model-vs-random barplot. Default: true.",
    )

    parser.add_argument(
        "--plot-top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values shown in the barplot. Default: 1 3 5.",
    )

    parser.add_argument(
        "--plot-formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help="Barplot output formats. Default: png pdf.",
    )

    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="DPI for raster plot formats such as PNG. Default: 300.",
    )

    parser.add_argument(
        "--plot-significance-correction",
        type=str,
        choices=["none", "bonferroni"],
        default="none",
        help=(
            "Correction used only for deciding which bars receive significance "
            "markers. Default: none. Use bonferroni for alpha / number_of_plotted_tests."
        ),
    )

    return parser.parse_args()


def check_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> list[str]:
    return [col for col in required_cols if col not in df.columns]


def hypergeometric_success_probability(
    *,
    n_population: int,
    n_hits: int,
    n_draws: int,
) -> float:
    """
    Probability of selecting at least one hit when drawing n_draws items
    without replacement from a population of n_population items containing n_hits hits.
    """

    if n_population <= 0 or n_draws <= 0 or n_hits <= 0:
        return 0.0

    k_eff = min(int(n_draws), int(n_population))
    n_hits = int(n_hits)
    n_population = int(n_population)

    if n_hits >= n_population:
        return 1.0

    n_misses = n_population - n_hits

    if k_eff > n_misses:
        return 1.0

    return 1.0 - (comb(n_misses, k_eff) / comb(n_population, k_eff))




def calc_bedroc_from_relevance(
    relevance: Iterable[bool | int | float],
    *,
    alpha: float = 80.0,
) -> float:
    """
    Calculate BEDROC from a ranked binary relevance vector.

    The input order must already be ranked from best to worst, here by
    descending attention weight. This implementation follows the same discrete
    RIE/BEDROC normalization used by RDKit's CalcBEDROC.
    """
    rel = [bool(x) for x in relevance]
    n_total = len(rel)

    if n_total == 0:
        return float("nan")

    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("BEDROC alpha must be greater than zero.")

    n_hits = int(sum(rel))
    if n_hits == 0:
        return 0.0
    if n_hits == n_total:
        return 1.0

    denominator = (1.0 / n_total) * (
        (1.0 - exp(-alpha)) / (exp(alpha / n_total) - 1.0)
    )

    weighted_sum = 0.0
    for idx, is_hit in enumerate(rel, start=1):
        if is_hit:
            weighted_sum += exp(-(alpha * idx) / n_total)

    rie = weighted_sum / (n_hits * denominator)
    hit_ratio = n_hits / n_total

    rie_max = (1.0 - exp(-alpha * hit_ratio)) / (
        hit_ratio * (1.0 - exp(-alpha))
    )
    rie_min = (1.0 - exp(alpha * hit_ratio)) / (
        hit_ratio * (1.0 - exp(alpha))
    )

    if rie_max == rie_min:
        return 1.0

    bedroc = (rie - rie_min) / (rie_max - rie_min)
    return float(max(0.0, min(1.0, bedroc)))


def calc_ndcg_at_k_from_relevance(
    relevance: Iterable[bool | int | float],
    *,
    k: int = 10,
) -> float:
    """
    Calculate nDCG@k from a ranked binary relevance vector.

    The input order must already be ranked from best to worst, here by
    descending attention weight. Relevance is binary: hit = 1, non-hit = 0.
    """
    rel = [1.0 if bool(x) else 0.0 for x in relevance]

    if not rel:
        return float("nan")

    k_eff = min(int(k), len(rel))
    if k_eff <= 0:
        return float("nan")

    def dcg(values: list[float]) -> float:
        return sum(value / log2(rank + 1) for rank, value in enumerate(values, start=1))

    observed = dcg(rel[:k_eff])
    ideal = dcg(sorted(rel, reverse=True)[:k_eff])

    if ideal == 0:
        return 0.0

    return float(observed / ideal)


def ranking_metric_columns(metric: str) -> tuple[str, str]:
    if metric == RMSD_METRIC:
        return "rmsd_bedroc", "rmsd_ndcg_at_k"
    if metric == NORM_O3A_METRIC:
        return "norm_o3a_bedroc", "norm_o3a_ndcg_at_k"
    raise ValueError(f"Unknown metric: {metric}")

def infer_series_label(
    bag_df: pd.DataFrame,
    series_col: str | None,
) -> tuple[object, object, object]:
    if not series_col or series_col not in bag_df.columns:
        return pd.NA, pd.NA, pd.NA

    values = bag_df[series_col].dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return UNKNOWN_SERIES, 0, UNKNOWN_SERIES

    unique_values = sorted(values.unique().tolist())
    observed = "|".join(unique_values)

    if len(unique_values) == 1:
        return unique_values[0], 1, observed

    main_label = values.value_counts().index[0]
    return str(main_label), len(unique_values), observed


def add_empty_topk_fields(
    row: dict,
    top_k_values: list[int],
    *,
    o3a_enabled: bool,
) -> dict:
    row["n_rmsd_hits_for_random_baseline"] = pd.NA
    row["n_rmsd_conformers_for_random_baseline"] = pd.NA
    row["rmsd_bedroc"] = pd.NA
    row["rmsd_ndcg_at_k"] = pd.NA

    if o3a_enabled:
        row["n_norm_o3a_hits_for_random_baseline"] = pd.NA
        row["n_norm_o3a_conformers_for_random_baseline"] = pd.NA
        row["norm_o3a_bedroc"] = pd.NA
        row["norm_o3a_ndcg_at_k"] = pd.NA

    for k in top_k_values:
        row[f"top_{k}_success"] = pd.NA
        row[f"top_{k}_best_rmsd_in_topk"] = pd.NA
        row[f"top_{k}_n_checked"] = 0
        row[f"top_{k}_random_success_probability"] = pd.NA

        if o3a_enabled:
            row[f"norm_o3a_top_{k}_success"] = pd.NA
            row[f"norm_o3a_top_{k}_best_norm_o3a_in_topk"] = pd.NA
            row[f"norm_o3a_top_{k}_n_checked"] = 0
            row[f"norm_o3a_top_{k}_random_success_probability"] = pd.NA

    return row


def evaluate_bag(
    bag_df: pd.DataFrame,
    *,
    file_name: str,
    bag_id: str,
    top_k_values: list[int],
    bedroc_alpha: float,
    ndcg_k: int,
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    attention_col: str,
    rmsd_col: str,
    min_rmsd_col: str,
    o3a_col: str,
    max_o3a_col: str,
    series_col: str | None,
    descending_attention: bool,
) -> dict:
    o3a_enabled = norm_o3a_threshold is not None

    series_label, n_unique_series, observed_series = infer_series_label(
        bag_df, series_col
    )

    result = {
        "file": file_name,
        "bag_id": bag_id,
        "series": series_label,
        "n_unique_series_in_bag": n_unique_series,
        "observed_series_in_bag": observed_series,
        "n_rows_raw": len(bag_df),
        "n_rows_used_rmsd": 0,
        "n_rows_used_norm_o3a": 0,
        "min_best_rmsd_to_exp": pd.NA,
        "max_norm_o3a_to_exp": pd.NA,
        "bedroc_alpha": bedroc_alpha,
        "ndcg_k": ndcg_k,
        "valid": False,
        "rmsd_valid": False,
        "norm_o3a_valid": False if o3a_enabled else pd.NA,
        "status": "ok",
        "rmsd_status": "ok",
        "norm_o3a_status": "not_evaluated" if not o3a_enabled else "ok",
    }

    result = add_empty_topk_fields(result, top_k_values, o3a_enabled=o3a_enabled)

    # RMSD evaluation
    missing_rmsd_cols = check_required_columns(
        bag_df, [attention_col, rmsd_col, min_rmsd_col]
    )

    if missing_rmsd_cols:
        result["status"] = "skipped_missing_rmsd_columns: " + ", ".join(
            missing_rmsd_cols
        )
        result["rmsd_status"] = result["status"]
    else:
        rmsd_work = bag_df.copy()
        rmsd_work[attention_col] = pd.to_numeric(
            rmsd_work[attention_col], errors="coerce"
        )
        rmsd_work[rmsd_col] = pd.to_numeric(rmsd_work[rmsd_col], errors="coerce")
        rmsd_work[min_rmsd_col] = pd.to_numeric(
            rmsd_work[min_rmsd_col], errors="coerce"
        )

        rmsd_work = rmsd_work.dropna(subset=[attention_col, rmsd_col, min_rmsd_col])
        result["n_rows_used_rmsd"] = len(rmsd_work)

        if rmsd_work.empty:
            result["status"] = "skipped_no_numeric_rmsd_required_values"
            result["rmsd_status"] = result["status"]
        else:
            min_rmsd = float(rmsd_work[min_rmsd_col].min())
            result["min_best_rmsd_to_exp"] = min_rmsd

            if min_rmsd > rmsd_threshold:
                result["rmsd_status"] = "invalid_min_rmsd_above_threshold"
            else:
                result["valid"] = True
                result["rmsd_valid"] = True

                ranked = rmsd_work.sort_values(
                    by=attention_col,
                    ascending=not descending_attention,
                    kind="mergesort",
                )

                n_population = len(ranked)
                n_hits = int((ranked[rmsd_col] <= rmsd_threshold).sum())

                result["n_rmsd_hits_for_random_baseline"] = n_hits
                result["n_rmsd_conformers_for_random_baseline"] = n_population

                rmsd_relevance = (ranked[rmsd_col] <= rmsd_threshold).tolist()
                result["rmsd_bedroc"] = calc_bedroc_from_relevance(
                    rmsd_relevance,
                    alpha=bedroc_alpha,
                )
                result["rmsd_ndcg_at_k"] = calc_ndcg_at_k_from_relevance(
                    rmsd_relevance,
                    k=ndcg_k,
                )

                for k in top_k_values:
                    k_eff = min(k, len(ranked))
                    topk = ranked.head(k_eff)

                    result[f"top_{k}_success"] = bool(
                        (topk[rmsd_col] <= rmsd_threshold).any()
                    )
                    result[f"top_{k}_best_rmsd_in_topk"] = float(topk[rmsd_col].min())
                    result[f"top_{k}_n_checked"] = k_eff
                    result[f"top_{k}_random_success_probability"] = (
                        hypergeometric_success_probability(
                            n_population=n_population,
                            n_hits=n_hits,
                            n_draws=k,
                        )
                    )

    # Normalized O3A evaluation
    if not o3a_enabled:
        return result

    missing_o3a_cols = check_required_columns(
        bag_df, [attention_col, o3a_col, max_o3a_col]
    )

    if missing_o3a_cols:
        result["norm_o3a_status"] = "skipped_missing_o3a_columns: " + ", ".join(
            missing_o3a_cols
        )
        return result

    o3a_work = bag_df.copy()
    o3a_work[attention_col] = pd.to_numeric(
        o3a_work[attention_col], errors="coerce"
    )
    o3a_work[o3a_col] = pd.to_numeric(o3a_work[o3a_col], errors="coerce")
    o3a_work[max_o3a_col] = pd.to_numeric(o3a_work[max_o3a_col], errors="coerce")

    o3a_work = o3a_work.dropna(subset=[attention_col, o3a_col, max_o3a_col])
    o3a_work = o3a_work[o3a_work[max_o3a_col] != 0].copy()

    result["n_rows_used_norm_o3a"] = len(o3a_work)

    if o3a_work.empty:
        result["norm_o3a_status"] = "skipped_no_numeric_o3a_required_values"
        return result

    o3a_work["norm_o3a_to_exp"] = o3a_work[o3a_col] / o3a_work[max_o3a_col]
    o3a_work = o3a_work.dropna(subset=["norm_o3a_to_exp"])

    if o3a_work.empty:
        result["norm_o3a_status"] = "skipped_no_valid_norm_o3a_values"
        return result

    max_norm_o3a = float(o3a_work["norm_o3a_to_exp"].max())
    result["max_norm_o3a_to_exp"] = max_norm_o3a

    if max_norm_o3a < float(norm_o3a_threshold):
        result["norm_o3a_status"] = "invalid_max_norm_o3a_below_threshold"
        return result

    result["valid"] = True
    result["norm_o3a_valid"] = True

    ranked = o3a_work.sort_values(
        by=attention_col,
        ascending=not descending_attention,
        kind="mergesort",
    )

    n_population = len(ranked)
    n_hits = int((ranked["norm_o3a_to_exp"] >= float(norm_o3a_threshold)).sum())

    result["n_norm_o3a_hits_for_random_baseline"] = n_hits
    result["n_norm_o3a_conformers_for_random_baseline"] = n_population

    norm_o3a_relevance = (
        ranked["norm_o3a_to_exp"] >= float(norm_o3a_threshold)
    ).tolist()
    result["norm_o3a_bedroc"] = calc_bedroc_from_relevance(
        norm_o3a_relevance,
        alpha=bedroc_alpha,
    )
    result["norm_o3a_ndcg_at_k"] = calc_ndcg_at_k_from_relevance(
        norm_o3a_relevance,
        k=ndcg_k,
    )

    for k in top_k_values:
        k_eff = min(k, len(ranked))
        topk = ranked.head(k_eff)

        result[f"norm_o3a_top_{k}_success"] = bool(
            (topk["norm_o3a_to_exp"] >= float(norm_o3a_threshold)).any()
        )
        result[f"norm_o3a_top_{k}_best_norm_o3a_in_topk"] = float(
            topk["norm_o3a_to_exp"].max()
        )
        result[f"norm_o3a_top_{k}_n_checked"] = k_eff
        result[f"norm_o3a_top_{k}_random_success_probability"] = (
            hypergeometric_success_probability(
                n_population=n_population,
                n_hits=n_hits,
                n_draws=k,
            )
        )

    return result


def get_metric_config(
    metric: str,
    *,
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
) -> dict:
    if metric == RMSD_METRIC:
        return {
            "metric": RMSD_METRIC,
            "threshold": rmsd_threshold,
            "valid_col": "rmsd_valid",
            "direction": "<=",
            "label": "RMSD",
            "value_label": "best_rmsd_to_exp",
        }

    if metric == NORM_O3A_METRIC:
        return {
            "metric": NORM_O3A_METRIC,
            "threshold": norm_o3a_threshold,
            "valid_col": "norm_o3a_valid",
            "direction": ">=",
            "label": "normalized O3A",
            "value_label": "norm_o3a_to_exp",
        }

    raise ValueError(f"Unknown metric: {metric}")


def success_column(metric: str, k: int) -> str:
    if metric == RMSD_METRIC:
        return f"top_{k}_success"
    if metric == NORM_O3A_METRIC:
        return f"norm_o3a_top_{k}_success"
    raise ValueError(f"Unknown metric: {metric}")


def random_baseline_column(metric: str, k: int) -> str:
    if metric == RMSD_METRIC:
        return f"top_{k}_random_success_probability"
    if metric == NORM_O3A_METRIC:
        return f"norm_o3a_top_{k}_random_success_probability"
    raise ValueError(f"Unknown metric: {metric}")


def safe_sum_boolean(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def safe_sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())



def paired_ttest_model_vs_random(
    valid: pd.DataFrame,
    *,
    model_success_col: str,
    random_probability_col: str,
    alternative: str,
) -> dict:
    """
    Paired one-sample t-test on per-bag differences:

        difference_i = model_success_i - random_success_probability_i

    where model_success_i is 0/1 and random_success_probability_i is the
    hypergeometric probability for the same bag and the same top-k value.

    H0: mean(difference) = 0
    H1 depends on --ttest-alternative:
        greater   -> mean(difference) > 0
        two-sided -> mean(difference) != 0
        less      -> mean(difference) < 0
    """

    if stats is None:
        return {
            "ttest_n": 0,
            "ttest_mean_difference": float("nan"),
            "ttest_std_difference": float("nan"),
            "ttest_t_statistic": float("nan"),
            "ttest_degrees_of_freedom": float("nan"),
            "ttest_pvalue": float("nan"),
            "ttest_alternative": alternative,
            "ttest_status": "not_computed_scipy_not_installed",
        }

    model = pd.to_numeric(valid[model_success_col], errors="coerce")
    random = pd.to_numeric(valid[random_probability_col], errors="coerce")

    paired = pd.DataFrame({"model": model, "random": random}).dropna()

    if paired.empty:
        return {
            "ttest_n": 0,
            "ttest_mean_difference": float("nan"),
            "ttest_std_difference": float("nan"),
            "ttest_t_statistic": float("nan"),
            "ttest_degrees_of_freedom": float("nan"),
            "ttest_pvalue": float("nan"),
            "ttest_alternative": alternative,
            "ttest_status": "not_computed_no_paired_values",
        }

    differences = paired["model"] - paired["random"]
    n = int(len(differences))
    mean_diff = float(differences.mean())
    std_diff = float(differences.std(ddof=1)) if n >= 2 else float("nan")

    n_model_better = int((differences > 0).sum())
    n_equal = int((differences == 0).sum())
    n_model_worse = int((differences < 0).sum())

    if n < 2:
        return {
            "ttest_n": n,
            "ttest_mean_difference": mean_diff,
            "ttest_std_difference": std_diff,
            "ttest_t_statistic": float("nan"),
            "ttest_degrees_of_freedom": float("nan"),
            "ttest_pvalue": float("nan"),
            "ttest_alternative": alternative,
            "ttest_n_model_better_than_random": n_model_better,
            "ttest_n_equal_to_random": n_equal,
            "ttest_n_model_worse_than_random": n_model_worse,
            "ttest_status": "not_computed_less_than_two_paired_values",
        }

    if std_diff == 0 or pd.isna(std_diff):
        return {
            "ttest_n": n,
            "ttest_mean_difference": mean_diff,
            "ttest_std_difference": std_diff,
            "ttest_t_statistic": float("nan"),
            "ttest_degrees_of_freedom": n - 1,
            "ttest_pvalue": float("nan"),
            "ttest_alternative": alternative,
            "ttest_n_model_better_than_random": n_model_better,
            "ttest_n_equal_to_random": n_equal,
            "ttest_n_model_worse_than_random": n_model_worse,
            "ttest_status": "not_computed_zero_variance_in_differences",
        }

    test = stats.ttest_1samp(
        differences.to_numpy(),
        popmean=0.0,
        alternative=alternative,
    )

    return {
        "ttest_n": n,
        "ttest_mean_difference": mean_diff,
        "ttest_std_difference": std_diff,
        "ttest_t_statistic": float(test.statistic),
        "ttest_degrees_of_freedom": n - 1,
        "ttest_pvalue": float(test.pvalue),
        "ttest_alternative": alternative,
        "ttest_n_model_better_than_random": n_model_better,
        "ttest_n_equal_to_random": n_equal,
        "ttest_n_model_worse_than_random": n_model_worse,
        "ttest_status": "ok",
    }


def format_pvalue(pvalue: float) -> str:
    if pd.isna(pvalue):
        return "NA"
    if pvalue < 1e-4:
        return f"{pvalue:.2e}"
    return f"{pvalue:.4f}"


def ttest_log_phrase(row: pd.Series, *, alpha: float) -> str:
    status = row.get("ttest_status", "not_available")

    if status != "ok":
        return f" Paired t-test not computed ({status})."

    pvalue = float(row["ttest_pvalue"])
    significant = pvalue < alpha
    significance_text = "significant" if significant else "not significant"

    return (
        " Paired t-test against random baseline: "
        f"mean(model - random) = {row['ttest_mean_difference']:.4f}, "
        f"t({int(row['ttest_degrees_of_freedom'])}) = "
        f"{row['ttest_t_statistic']:.3f}, "
        f"p = {format_pvalue(pvalue)} "
        f"({significance_text} at alpha = {alpha})."
    )


def make_global_summary(
    *,
    per_bag: pd.DataFrame,
    top_k_values: list[int],
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    ttest_alternative: str,
) -> pd.DataFrame:
    metrics = [RMSD_METRIC]
    if norm_o3a_threshold is not None:
        metrics.append(NORM_O3A_METRIC)

    rows = []
    n_total_bags = len(per_bag)

    for metric in metrics:
        cfg = get_metric_config(
            metric,
            rmsd_threshold=rmsd_threshold,
            norm_o3a_threshold=norm_o3a_threshold,
        )

        valid = per_bag[per_bag[cfg["valid_col"]] == True].copy()
        n_valid_bags = len(valid)

        for k in top_k_values:
            col = success_column(metric, k)
            random_col = random_baseline_column(metric, k)

            n_success = safe_sum_boolean(valid[col]) if n_valid_bags > 0 else 0

            success_rate = (
                n_success / n_valid_bags if n_valid_bags > 0 else float("nan")
            )
            success_rate_percent = (
                success_rate * 100 if pd.notna(success_rate) else float("nan")
            )

            random_expected_success_bags = (
                safe_sum_float(valid[random_col]) if n_valid_bags > 0 else float("nan")
            )
            random_success_rate = (
                random_expected_success_bags / n_valid_bags
                if n_valid_bags > 0
                else float("nan")
            )
            random_success_rate_percent = (
                random_success_rate * 100
                if pd.notna(random_success_rate)
                else float("nan")
            )

            ttest_result = paired_ttest_model_vs_random(
                valid,
                model_success_col=col,
                random_probability_col=random_col,
                alternative=ttest_alternative,
            )

            rows.append(
                {
                    "metric": metric,
                    "top_k": k,
                    "threshold": cfg["threshold"],
                    "threshold_direction": cfg["direction"],
                    "n_total_bags": n_total_bags,
                    "n_valid_bags": n_valid_bags,
                    "n_invalid_or_skipped_bags": n_total_bags - n_valid_bags,
                    "n_success_bags": n_success,
                    "success_rate": success_rate,
                    "success_rate_percent": success_rate_percent,
                    "random_baseline_expected_success_bags": random_expected_success_bags,
                    "random_baseline_success_rate": random_success_rate,
                    "random_baseline_success_rate_percent": random_success_rate_percent,
                    "calculation": (
                        f"{n_success} successful valid bags / {n_valid_bags} "
                        f"valid bags * 100 = {success_rate_percent:.2f}%"
                        if n_valid_bags > 0
                        else "0 valid bags; success rate undefined"
                    ),
                    "random_baseline_calculation": (
                        f"Mean hypergeometric probability over {n_valid_bags} "
                        f"valid bags = {random_success_rate_percent:.2f}%; "
                        f"expected successful bags = "
                        f"{random_expected_success_bags:.2f} / {n_valid_bags}"
                        if n_valid_bags > 0
                        else "0 valid bags; random baseline undefined"
                    ),
                    **ttest_result,
                }
            )

    return pd.DataFrame(rows)


def make_series_reports(
    *,
    per_bag: pd.DataFrame,
    top_k_values: list[int],
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    ttest_alternative: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = per_bag[per_bag["series"].notna()].copy()

    if usable.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = [RMSD_METRIC]
    if norm_o3a_threshold is not None:
        metrics.append(NORM_O3A_METRIC)

    overview_rows = []
    summary_rows = []

    for metric in metrics:
        cfg = get_metric_config(
            metric,
            rmsd_threshold=rmsd_threshold,
            norm_o3a_threshold=norm_o3a_threshold,
        )

        for series, group in usable.groupby("series", dropna=False, sort=True):
            n_total = len(group)
            n_valid = int((group[cfg["valid_col"]] == True).sum())

            overview_rows.append(
                {
                    "metric": metric,
                    "series": series,
                    "threshold": cfg["threshold"],
                    "threshold_direction": cfg["direction"],
                    "n_total_bags": n_total,
                    "n_valid_bags": n_valid,
                    "n_invalid_or_skipped_bags": n_total - n_valid,
                    "valid_fraction": n_valid / n_total if n_total > 0 else float("nan"),
                    "valid_fraction_percent": (
                        n_valid / n_total * 100 if n_total > 0 else float("nan")
                    ),
                }
            )

        valid = usable[usable[cfg["valid_col"]] == True].copy()

        for series, group in valid.groupby("series", dropna=False, sort=True):
            n_valid = len(group)

            for k in top_k_values:
                col = success_column(metric, k)
                random_col = random_baseline_column(metric, k)

                n_success = safe_sum_boolean(group[col]) if n_valid > 0 else 0

                success_rate = (
                    n_success / n_valid if n_valid > 0 else float("nan")
                )
                success_rate_percent = (
                    success_rate * 100 if pd.notna(success_rate) else float("nan")
                )

                random_expected_success_bags = (
                    safe_sum_float(group[random_col]) if n_valid > 0 else float("nan")
                )
                random_success_rate = (
                    random_expected_success_bags / n_valid
                    if n_valid > 0
                    else float("nan")
                )
                random_success_rate_percent = (
                    random_success_rate * 100
                    if pd.notna(random_success_rate)
                    else float("nan")
                )

                ttest_result = paired_ttest_model_vs_random(
                    group,
                    model_success_col=col,
                    random_probability_col=random_col,
                    alternative=ttest_alternative,
                )

                summary_rows.append(
                    {
                        "metric": metric,
                        "series": series,
                        "top_k": k,
                        "threshold": cfg["threshold"],
                        "threshold_direction": cfg["direction"],
                        "n_valid_bags": n_valid,
                        "n_success_bags": n_success,
                        "success_rate": success_rate,
                        "success_rate_percent": success_rate_percent,
                        "random_baseline_expected_success_bags": random_expected_success_bags,
                        "random_baseline_success_rate": random_success_rate,
                        "random_baseline_success_rate_percent": random_success_rate_percent,
                        "calculation": (
                            f"{n_success} successful valid bags / {n_valid} "
                            f"valid bags * 100 = {success_rate_percent:.2f}%"
                            if n_valid > 0
                            else "0 valid bags; success rate undefined"
                        ),
                        "random_baseline_calculation": (
                            f"Mean hypergeometric probability over {n_valid} "
                            f"valid bags = {random_success_rate_percent:.2f}%; "
                            f"expected successful bags = "
                            f"{random_expected_success_bags:.2f} / {n_valid}"
                            if n_valid > 0
                            else "0 valid bags; random baseline undefined"
                        ),
                        **ttest_result,
                    }
                )

    series_overview = pd.DataFrame(overview_rows)
    series_summary = pd.DataFrame(summary_rows)

    if not series_overview.empty:
        series_overview = series_overview.sort_values(
            by=["metric", "n_valid_bags", "n_total_bags", "series"],
            ascending=[True, False, False, True],
        )

    if not series_summary.empty:
        series_summary = series_summary.sort_values(
            by=[
                "metric",
                "top_k",
                "success_rate",
                "random_baseline_success_rate",
                "n_success_bags",
                "n_valid_bags",
                "series",
            ],
            ascending=[True, True, False, True, False, False, True],
        )

    return series_overview, series_summary




def summarize_numeric_series(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if len(numeric) >= 2 else float("nan"),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def make_ranking_metrics_summary(
    *,
    per_bag: pd.DataFrame,
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    bedroc_alpha: float,
    ndcg_k: int,
) -> pd.DataFrame:
    metrics = [RMSD_METRIC]
    if norm_o3a_threshold is not None:
        metrics.append(NORM_O3A_METRIC)

    rows = []

    for metric in metrics:
        cfg = get_metric_config(
            metric,
            rmsd_threshold=rmsd_threshold,
            norm_o3a_threshold=norm_o3a_threshold,
        )
        bedroc_col, ndcg_col = ranking_metric_columns(metric)
        valid = per_bag[per_bag[cfg["valid_col"]] == True].copy()

        bedroc_stats = summarize_numeric_series(valid[bedroc_col]) if bedroc_col in valid else summarize_numeric_series(pd.Series(dtype=float))
        ndcg_stats = summarize_numeric_series(valid[ndcg_col]) if ndcg_col in valid else summarize_numeric_series(pd.Series(dtype=float))

        rows.append(
            {
                "metric": metric,
                "threshold": cfg["threshold"],
                "threshold_direction": cfg["direction"],
                "n_valid_bags": int(len(valid)),
                "bedroc_alpha": bedroc_alpha,
                "bedroc_mean": bedroc_stats["mean"],
                "bedroc_std": bedroc_stats["std"],
                "bedroc_median": bedroc_stats["median"],
                "bedroc_min": bedroc_stats["min"],
                "bedroc_max": bedroc_stats["max"],
                "ndcg_k": ndcg_k,
                "ndcg_mean": ndcg_stats["mean"],
                "ndcg_std": ndcg_stats["std"],
                "ndcg_median": ndcg_stats["median"],
                "ndcg_min": ndcg_stats["min"],
                "ndcg_max": ndcg_stats["max"],
            }
        )

    return pd.DataFrame(rows)


def make_series_ranking_metrics_summary(
    *,
    per_bag: pd.DataFrame,
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    bedroc_alpha: float,
    ndcg_k: int,
) -> pd.DataFrame:
    usable = per_bag[per_bag["series"].notna()].copy()

    if usable.empty:
        return pd.DataFrame()

    metrics = [RMSD_METRIC]
    if norm_o3a_threshold is not None:
        metrics.append(NORM_O3A_METRIC)

    rows = []

    for metric in metrics:
        cfg = get_metric_config(
            metric,
            rmsd_threshold=rmsd_threshold,
            norm_o3a_threshold=norm_o3a_threshold,
        )
        bedroc_col, ndcg_col = ranking_metric_columns(metric)
        valid = usable[usable[cfg["valid_col"]] == True].copy()

        for series, group in valid.groupby("series", dropna=False, sort=True):
            bedroc_stats = summarize_numeric_series(group[bedroc_col]) if bedroc_col in group else summarize_numeric_series(pd.Series(dtype=float))
            ndcg_stats = summarize_numeric_series(group[ndcg_col]) if ndcg_col in group else summarize_numeric_series(pd.Series(dtype=float))

            rows.append(
                {
                    "metric": metric,
                    "series": series,
                    "threshold": cfg["threshold"],
                    "threshold_direction": cfg["direction"],
                    "n_valid_bags": int(len(group)),
                    "bedroc_alpha": bedroc_alpha,
                    "bedroc_mean": bedroc_stats["mean"],
                    "bedroc_std": bedroc_stats["std"],
                    "bedroc_median": bedroc_stats["median"],
                    "bedroc_min": bedroc_stats["min"],
                    "bedroc_max": bedroc_stats["max"],
                    "ndcg_k": ndcg_k,
                    "ndcg_mean": ndcg_stats["mean"],
                    "ndcg_std": ndcg_stats["std"],
                    "ndcg_median": ndcg_stats["median"],
                    "ndcg_min": ndcg_stats["min"],
                    "ndcg_max": ndcg_stats["max"],
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["metric", "bedroc_mean", "ndcg_mean", "n_valid_bags", "series"],
            ascending=[True, False, False, False, True],
        )

    return out


def make_ranking_metrics_log_lines(
    *,
    ranking_summary: pd.DataFrame,
    series_ranking_summary: pd.DataFrame,
) -> list[str]:
    lines = [
        "",
        "Ranking metrics",
        "-" * 44,
    ]

    if ranking_summary.empty:
        lines.append("Ranking metrics were not computed because no valid bags were available.")
        return lines

    for _, row in ranking_summary.iterrows():
        metric_label = "RMSD" if row["metric"] == RMSD_METRIC else "normalized O3A"
        lines.append(
            f"{metric_label}: mean BEDROC(alpha={row['bedroc_alpha']:.3g}) = "
            f"{row['bedroc_mean']:.4f}; mean nDCG@{int(row['ndcg_k'])} = "
            f"{row['ndcg_mean']:.4f} across {int(row['n_valid_bags'])} valid bags."
        )

    if not series_ranking_summary.empty:
        lines.append("")
        lines.append("Best-performing series by mean BEDROC:")
        for metric in series_ranking_summary["metric"].unique():
            block = series_ranking_summary[series_ranking_summary["metric"] == metric].copy()
            if block.empty:
                continue
            best = block.sort_values(
                by=["bedroc_mean", "ndcg_mean", "n_valid_bags", "series"],
                ascending=[False, False, False, True],
            ).iloc[0]
            metric_label = "RMSD" if metric == RMSD_METRIC else "normalized O3A"
            lines.append(
                f"- {metric_label}: {best['series']} "
                f"(mean BEDROC={best['bedroc_mean']:.4f}, "
                f"mean nDCG@{int(best['ndcg_k'])}={best['ndcg_mean']:.4f}, "
                f"n={int(best['n_valid_bags'])})."
            )

    return lines

def make_log_lines(
    *,
    summary: pd.DataFrame,
    series_overview: pd.DataFrame,
    series_summary: pd.DataFrame,
    n_total_bags: int,
    rmsd_threshold: float,
    norm_o3a_threshold: float | None,
    attention_col: str,
    rmsd_col: str,
    min_rmsd_col: str,
    o3a_col: str,
    max_o3a_col: str,
    series_col: str | None,
    series_reporting_enabled: bool,
    ttest_alternative: str,
    ttest_alpha: float,
) -> list[str]:
    lines = [
        "Top-k attention success calculation log",
        "=" * 44,
        f"Total bags evaluated: {n_total_bags}",
        "",
        "Random baseline",
        "-" * 44,
        (
            "For each eligible bag, the hypergeometric random baseline estimates "
            "the probability of retrieving at least one successful conformer when "
            "randomly selecting k conformers without replacement."
        ),
        "Formula: P(success) = 1 - C(N - H, k) / C(N, k)",
        "N = number of conformers in the eligible bag.",
        "H = number of successful conformers in the eligible bag.",
        "If k is larger than bag size, k is replaced by the bag size.",
        "",
        "Paired t-test",
        "-" * 44,
        (
            "The t-test is computed over valid bags using per-bag differences: "
            "model_success - random_success_probability."
        ),
        f"Alternative hypothesis: {ttest_alternative}.",
        f"Reporting alpha: {ttest_alpha}.",
        "",
        "RMSD metric",
        "-" * 44,
        f"RMSD threshold: {rmsd_threshold}",
        f"Validity rule: a bag is RMSD-valid if min({min_rmsd_col}) <= {rmsd_threshold}",
        (
            "Success rule: for each RMSD-valid bag, success at top-k is counted if "
            f"at least one conformer among the top-k ranked by {attention_col} "
            f"has {rmsd_col} <= {rmsd_threshold}"
        ),
        "",
    ]

    if norm_o3a_threshold is not None:
        lines.extend(
            [
                "Normalized O3A metric",
                "-" * 44,
                f"Normalized O3A threshold: {norm_o3a_threshold}",
                f"Normalization rule: norm_o3a_to_exp = {o3a_col} / {max_o3a_col}",
                (
                    "Validity rule: a bag is normalized-O3A-valid if "
                    f"max(norm_o3a_to_exp) >= {norm_o3a_threshold}"
                ),
                (
                    "Success rule: for each normalized-O3A-valid bag, success at top-k "
                    f"is counted if at least one conformer among the top-k ranked by "
                    f"{attention_col} has norm_o3a_to_exp >= {norm_o3a_threshold}"
                ),
                "",
            ]
        )

    for metric in summary["metric"].unique():
        block = summary[summary["metric"] == metric]

        metric_label = "RMSD" if metric == RMSD_METRIC else "normalized O3A"
        value_label = rmsd_col if metric == RMSD_METRIC else "norm_o3a_to_exp"
        direction = "<=" if metric == RMSD_METRIC else ">="

        lines.append(f"{metric_label} top-k results:")

        for _, row in block.iterrows():
            k = int(row["top_k"])
            threshold = row["threshold"]
            n_valid = int(row["n_valid_bags"])
            n_success = int(row["n_success_bags"])

            if n_valid == 0:
                lines.append(
                    f"Top-{k}: for {metric_label} threshold {threshold}, "
                    "there are 0 valid bags; success rate is undefined. "
                    "Random baseline is also undefined."
                )
            else:
                lines.append(
                    f"Top-{k}: for {metric_label} threshold {threshold}, "
                    f"there are {n_valid} valid bags. "
                    f"For {n_success} bags, at least one conformer among the top-{k} "
                    f"attention-ranked conformers has {value_label} {direction} "
                    f"{threshold}. Therefore, {n_success} / {n_valid} * 100 = "
                    f"{row['success_rate_percent']:.2f}%. "
                    f"Hypergeometric random baseline: expected "
                    f"{row['random_baseline_expected_success_bags']:.2f} / "
                    f"{n_valid} successful bags = "
                    f"{row['random_baseline_success_rate_percent']:.2f}%."
                    + ttest_log_phrase(row, alpha=ttest_alpha)
                )

        lines.append("")

    if not series_reporting_enabled:
        lines.append("Series reporting: disabled or no series column was found.")
        return lines

    if series_overview.empty:
        lines.append(
            f"Series reporting: column '{series_col}' was found, "
            "but no usable series labels were detected."
        )
        return lines

    series_names = sorted(series_overview["series"].astype(str).unique().tolist())

    lines.extend(
        [
            f"Series reporting column: {series_col}",
            f"Series met: {', '.join(series_names)}",
            "",
            "Entries per series:",
        ]
    )

    for series in series_names:
        rows = series_overview[series_overview["series"].astype(str) == series]
        total_bags = int(rows["n_total_bags"].max()) if not rows.empty else 0

        parts = []
        for _, row in rows.iterrows():
            metric_label = "RMSD" if row["metric"] == RMSD_METRIC else "normalized O3A"
            parts.append(
                f"{metric_label}: valid bags = {int(row['n_valid_bags'])}, "
                f"invalid/skipped bags = {int(row['n_invalid_or_skipped_bags'])}"
            )

        lines.append(
            f"- {series}: total bags = {total_bags}; " + "; ".join(parts) + "."
        )

    lines.append("")

    if series_summary.empty:
        lines.append(
            "No valid bags with series labels were available for series-level success rates."
        )
        return lines

    def format_series_performance(row: pd.Series) -> str:
        observed = float(row["success_rate_percent"])
        random = float(row["random_baseline_success_rate_percent"])
        delta = observed - random
        pvalue = row.get("ttest_pvalue", float("nan"))
        pvalue_num = pd.to_numeric(pd.Series([pvalue]), errors="coerce").iloc[0]

        if pd.isna(pvalue_num):
            sig_text = "p=NA"
        elif pvalue_num < ttest_alpha:
            sig_text = f"p={format_pvalue(pvalue_num)}, significant"
        else:
            sig_text = f"p={format_pvalue(pvalue_num)}, not significant"

        return (
            f"{row['series']} ("
            f"{int(row['n_success_bags'])}/{int(row['n_valid_bags'])}, "
            f"observed {observed:.2f}%, "
            f"random {random:.2f}%, "
            f"delta {delta:+.2f} pp, "
            f"{sig_text})"
        )

    lines.append("All series performance by top-k success rate:")

    for metric in series_summary["metric"].unique():
        metric_block = series_summary[series_summary["metric"] == metric]
        metric_label = "RMSD" if metric == RMSD_METRIC else "normalized O3A"

        lines.append(f"{metric_label}:")

        for k in sorted(metric_block["top_k"].unique()):
            block = metric_block[metric_block["top_k"] == k].copy()

            # Sort by observed success first, then by improvement over random baseline,
            # then by the number of valid bags. This keeps strong but reliable series
            # near the beginning while still reporting every series.
            block["delta_success_rate_percent"] = (
                pd.to_numeric(block["success_rate_percent"], errors="coerce")
                - pd.to_numeric(block["random_baseline_success_rate_percent"], errors="coerce")
            )
            block = block.sort_values(
                by=[
                    "success_rate_percent",
                    "delta_success_rate_percent",
                    "n_valid_bags",
                    "series",
                ],
                ascending=[False, False, False, True],
            )

            series_strings = [
                format_series_performance(row)
                for _, row in block.iterrows()
            ]

            lines.append(f"- Top-{int(k)}: {', '.join(series_strings)}")

    lines.append("")
    lines.append("Best-performing series by top-k success rate:")

    for metric in series_summary["metric"].unique():
        metric_block = series_summary[series_summary["metric"] == metric]
        metric_label = "RMSD" if metric == RMSD_METRIC else "normalized O3A"

        lines.append(f"{metric_label}:")

        for k in sorted(metric_block["top_k"].unique()):
            block = metric_block[metric_block["top_k"] == k].copy()
            best_rate = block["success_rate"].max()
            best = block[block["success_rate"] == best_rate].copy()

            best["delta_success_rate_percent"] = (
                pd.to_numeric(best["success_rate_percent"], errors="coerce")
                - pd.to_numeric(best["random_baseline_success_rate_percent"], errors="coerce")
            )
            best = best.sort_values(
                by=[
                    "n_success_bags",
                    "n_valid_bags",
                    "delta_success_rate_percent",
                    "random_baseline_success_rate",
                    "series",
                ],
                ascending=[False, False, False, True, True],
            )

            best_strings = [
                format_series_performance(row)
                for _, row in best.iterrows()
            ]

            lines.append(f"- Top-{int(k)}: {', '.join(best_strings)}")

    return lines



def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def interpolate_color(
    start_hex: str,
    end_hex: str,
    fraction: float,
) -> tuple[float, float, float]:
    start = hex_to_rgb01(start_hex)
    end = hex_to_rgb01(end_hex)
    fraction = max(0.0, min(1.0, float(fraction)))
    return tuple(start_i + (end_i - start_i) * fraction for start_i, end_i in zip(start, end))


def significance_marker(pvalue: float, *, alpha: float) -> str:
    """Return a single significance marker for any p-value below alpha."""
    if pd.isna(pvalue) or pvalue >= alpha:
        return ""
    return "*"


def make_topk_barplot(
    *,
    summary: pd.DataFrame,
    output_prefix: Path,
    plot_top_k_values: list[int],
    alpha: float,
    significance_correction: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """
    Create a grouped barplot comparing model top-k success rate with the
    hypergeometric random baseline.

    Bars are grouped by metric and top-k. Statistical significance is marked
    above the model bar using the paired t-test p-value already stored in the
    summary table.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover
        print(f"Barplot skipped because matplotlib is not installed: {exc}")
        return []

    if summary.empty:
        print("Barplot skipped because the summary table is empty.")
        return []

    requested_top_k = [int(k) for k in plot_top_k_values]
    metric_order = [RMSD_METRIC, NORM_O3A_METRIC]
    metric_labels = {
        RMSD_METRIC: "RMSD",
        NORM_O3A_METRIC: "norm. O3A",
    }

    plot_rows = []
    for metric in metric_order:
        for k in requested_top_k:
            row = summary[(summary["metric"] == metric) & (summary["top_k"] == k)]
            if row.empty:
                continue
            plot_rows.append(row.iloc[0])

    if not plot_rows:
        print(
            "Barplot skipped because none of --plot-top-k values are present "
            "in the summary. Make sure --top-k includes the same values."
        )
        return []

    plot_df = pd.DataFrame(plot_rows).reset_index(drop=True)

    pvalues = pd.to_numeric(plot_df["ttest_pvalue"], errors="coerce").dropna()
    if significance_correction == "bonferroni" and len(pvalues) > 0:
        alpha_used = alpha / len(pvalues)
        correction_label = f"Bonferroni-corrected α={alpha_used:.4g}"
    else:
        alpha_used = alpha
        correction_label = f"α={alpha_used:.3g}"

    n_groups = len(plot_df)
    x_positions = list(range(n_groups))
    bar_width = 0.34

    fig_width = max(8.5, 1.25 * n_groups + 2.5)
    fig_height = 5.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    model_colors = []
    random_colors = []
    metric_counts = plot_df["metric"].value_counts().to_dict()
    metric_seen = {RMSD_METRIC: 0, NORM_O3A_METRIC: 0}

    for _, row in plot_df.iterrows():
        metric = row["metric"]
        metric_seen[metric] += 1
        denominator = max(metric_counts.get(metric, 1) - 1, 1)
        frac = (metric_seen[metric] - 1) / denominator

        if metric == RMSD_METRIC:
            model_colors.append(interpolate_color("#D8ECF6", "#8CBFD9", frac))
            random_colors.append(interpolate_color("#F1F1F1", "#D6D6D6", frac))
        else:
            model_colors.append(interpolate_color("#FDE3CF", "#F5B78A", frac))
            random_colors.append(interpolate_color("#F1F1F1", "#D6D6D6", frac))

    model_values = pd.to_numeric(
        plot_df["success_rate_percent"], errors="coerce"
    ).to_numpy()
    random_values = pd.to_numeric(
        plot_df["random_baseline_success_rate_percent"], errors="coerce"
    ).to_numpy()

    model_x = [x - bar_width / 2 for x in x_positions]
    random_x = [x + bar_width / 2 for x in x_positions]

    ax.bar(
        model_x,
        model_values,
        width=bar_width,
        color=model_colors,
        edgecolor="#303030",
        linewidth=1.7,
        label="Model",
        zorder=3,
    )
    ax.bar(
        random_x,
        random_values,
        width=bar_width,
        color=random_colors,
        edgecolor="#303030",
        linewidth=1.7,
        label="Random baseline",
        zorder=3,
    )

    y_max = 0.0
    for value in list(model_values) + list(random_values):
        if pd.notna(value):
            y_max = max(y_max, float(value))
    y_top = min(118.0, max(100.0, y_max + 18.0))

    for i, row in plot_df.iterrows():
        pvalue = pd.to_numeric(pd.Series([row.get("ttest_pvalue")]), errors="coerce").iloc[0]
        marker = significance_marker(pvalue, alpha=alpha_used)
        if not marker:
            continue

        x1 = model_x[i]
        x2 = random_x[i]
        y = max(model_values[i], random_values[i]) + 4.0
        h = 2.0

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="#303030",
            linewidth=1.6,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + h + 0.8,
            marker,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#303030",
        )
        y_top = max(y_top, y + h + 8.0)

    x_labels = [
        f"{metric_labels.get(row['metric'], row['metric'])}\nTop-{int(row['top_k'])}"
        for _, row in plot_df.iterrows()
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Recovery Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, y_top)
    ax.set_title(
        "Top-k Recovery Rates vs Random Baseline",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=1.0, alpha=0.35, zorder=0)
    ax.xaxis.grid(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.6)
        ax.spines[spine].set_color("#303030")
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="both", width=1.5, length=5, color="#303030")

    legend_handles = [
        Patch(
            facecolor="#B7D7E8",
            edgecolor="#303030",
            linewidth=1.5,
            label="Model, RMSD",
        ),
        Patch(
            facecolor="#F5B78A",
            edgecolor="#303030",
            linewidth=1.5,
            label="Model, norm. O3A",
        ),
        Patch(
            facecolor="#E2E2E2",
            edgecolor="#303030",
            linewidth=1.5,
            label="Random baseline",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        fontsize=11,
    )

    ax.text(
        1.0,
        -0.18,
        f"Significance: paired t-test (MIL Model vs Random Baseline); {correction_label}. "
        "Marker: * p<α.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#404040",
    )

    fig.tight_layout()

    output_paths = []
    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        if not clean_fmt:
            continue
        out_path = output_prefix.with_name(
            f"{output_prefix.name}_barplot.{clean_fmt}"
        )
        save_kwargs = {"bbox_inches": "tight"}
        if clean_fmt in {"png", "jpg", "jpeg", "tiff", "tif"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def format_float_for_log(value: object, *, digits: int = 4) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def make_representative_rows(
    bag_df: pd.DataFrame,
    *,
    file_name: str,
    bag_id: str,
    representative_top_k_values: list[int],
    rmsd_threshold: float,
    attention_col: str,
    rmsd_col: str,
    min_rmsd_col: str,
    instance_id_col: str | None,
    series_col: str | None,
    descending_attention: bool,
) -> list[dict]:
    """
    Extract top-attended conformers for each representative bag.

    The output is long-form: one row per bag, top-N setting, and attention rank.
    Bioactive-like is defined only by RMSD:
        best_rmsd_to_exp <= rmsd_threshold
    """

    top_values = sorted(set(int(k) for k in representative_top_k_values if int(k) > 0))
    if not top_values:
        return []

    series_label, n_unique_series, observed_series = infer_series_label(
        bag_df, series_col
    )

    required = [attention_col, rmsd_col]
    missing = check_required_columns(bag_df, required)
    if missing:
        return [
            {
                "series": series_label,
                "file": file_name,
                "bag_id": bag_id,
                "status": "skipped_missing_columns: " + ", ".join(missing),
                "top_n": pd.NA,
                "attention_rank": pd.NA,
                "instance_id": pd.NA,
                "attention_weight": pd.NA,
                "best_rmsd_to_exp": pd.NA,
                "rmsd_threshold": rmsd_threshold,
                "rmsd_bioactive_like": pd.NA,
                "bag_min_best_rmsd_to_exp": pd.NA,
                "bag_rmsd_valid": pd.NA,
                "n_unique_series_in_bag": n_unique_series,
                "observed_series_in_bag": observed_series,
            }
        ]

    work = bag_df.copy()
    work[attention_col] = pd.to_numeric(work[attention_col], errors="coerce")
    work[rmsd_col] = pd.to_numeric(work[rmsd_col], errors="coerce")

    if min_rmsd_col in work.columns:
        work[min_rmsd_col] = pd.to_numeric(work[min_rmsd_col], errors="coerce")
        bag_min_rmsd = work[min_rmsd_col].dropna().min()
    else:
        bag_min_rmsd = work[rmsd_col].dropna().min()

    bag_min_rmsd = float(bag_min_rmsd) if pd.notna(bag_min_rmsd) else float("nan")
    bag_rmsd_valid = bool(bag_min_rmsd <= rmsd_threshold) if pd.notna(bag_min_rmsd) else pd.NA

    work = work.dropna(subset=[attention_col, rmsd_col]).copy()

    if work.empty:
        return [
            {
                "series": series_label,
                "file": file_name,
                "bag_id": bag_id,
                "status": "skipped_no_numeric_attention_or_rmsd_values",
                "top_n": pd.NA,
                "attention_rank": pd.NA,
                "instance_id": pd.NA,
                "attention_weight": pd.NA,
                "best_rmsd_to_exp": pd.NA,
                "rmsd_threshold": rmsd_threshold,
                "rmsd_bioactive_like": pd.NA,
                "bag_min_best_rmsd_to_exp": bag_min_rmsd,
                "bag_rmsd_valid": bag_rmsd_valid,
                "n_unique_series_in_bag": n_unique_series,
                "observed_series_in_bag": observed_series,
            }
        ]

    work = work.reset_index(drop=False).rename(columns={"index": "original_row_index"})

    if instance_id_col and instance_id_col in work.columns:
        work["_representative_instance_id"] = work[instance_id_col].astype(str)
    else:
        work["_representative_instance_id"] = (
            "row_" + work["original_row_index"].astype(str)
        )

    ranked = work.sort_values(
        by=attention_col,
        ascending=not descending_attention,
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict] = []
    for top_n in top_values:
        top_df = ranked.head(min(top_n, len(ranked)))
        for rank_idx, (_, row) in enumerate(top_df.iterrows(), start=1):
            rmsd_value = float(row[rmsd_col])
            rows.append(
                {
                    "series": series_label,
                    "file": file_name,
                    "bag_id": bag_id,
                    "status": "ok",
                    "top_n": top_n,
                    "attention_rank": rank_idx,
                    "instance_id": row["_representative_instance_id"],
                    "attention_weight": float(row[attention_col]),
                    "best_rmsd_to_exp": rmsd_value,
                    "rmsd_threshold": rmsd_threshold,
                    "rmsd_bioactive_like": bool(rmsd_value <= rmsd_threshold),
                    "bag_min_best_rmsd_to_exp": bag_min_rmsd,
                    "bag_rmsd_valid": bag_rmsd_valid,
                    "n_unique_series_in_bag": n_unique_series,
                    "observed_series_in_bag": observed_series,
                }
            )

    return rows


def format_representative_conformer_for_log(row: pd.Series) -> str:
    flag = "bioactive-like" if bool(row.get("rmsd_bioactive_like")) else "not bioactive-like"
    return (
        f"rank {int(row['attention_rank'])}: {row['instance_id']} "
        f"(attention={format_float_for_log(row['attention_weight'], digits=6)}, "
        f"RMSD={format_float_for_log(row['best_rmsd_to_exp'], digits=3)}, "
        f"{flag})"
    )


def make_representatives_log_lines(
    *,
    representatives: pd.DataFrame,
    rmsd_threshold: float,
) -> list[str]:
    lines = [
        "",
        "Series representatives and top-attended conformers",
        "-" * 44,
        f"Bioactive-like definition: best_rmsd_to_exp <= {rmsd_threshold}.",
    ]

    if representatives.empty:
        lines.append("No representative conformer rows were generated.")
        return lines

    ok = representatives[representatives["status"] == "ok"].copy()
    if ok.empty:
        lines.append("No valid representative conformer rows were available.")
        return lines

    for series in sorted(ok["series"].astype(str).unique().tolist()):
        series_block = ok[ok["series"].astype(str) == series].copy()
        lines.append(f"{series}:")

        bag_ids = sorted(series_block["bag_id"].astype(str).unique().tolist())
        for bag_id in bag_ids:
            bag_block = series_block[series_block["bag_id"].astype(str) == bag_id].copy()
            file_name = str(bag_block["file"].iloc[0])
            bag_min_rmsd = bag_block["bag_min_best_rmsd_to_exp"].iloc[0]
            bag_valid = bag_block["bag_rmsd_valid"].iloc[0]
            bag_valid_text = "valid" if bool(bag_valid) else "not valid"

            lines.append(
                f"  - {bag_id} ({file_name}; min RMSD="
                f"{format_float_for_log(bag_min_rmsd, digits=3)}, {bag_valid_text}):"
            )

            for top_n in sorted(bag_block["top_n"].dropna().astype(int).unique().tolist()):
                top_block = bag_block[bag_block["top_n"].astype(int) == top_n].copy()
                top_block = top_block.sort_values("attention_rank")
                formatted = "; ".join(
                    format_representative_conformer_for_log(row)
                    for _, row in top_block.iterrows()
                )
                lines.append(f"    Top-{top_n}: {formatted}")

    return lines


def make_error_row(
    *,
    file_name: str,
    bag_id: str,
    status: str,
    top_k_values: list[int],
    n_rows_raw: int | object,
    o3a_enabled: bool,
) -> dict:
    row = {
        "file": file_name,
        "bag_id": bag_id,
        "series": pd.NA,
        "n_unique_series_in_bag": pd.NA,
        "observed_series_in_bag": pd.NA,
        "n_rows_raw": n_rows_raw,
        "n_rows_used_rmsd": 0,
        "n_rows_used_norm_o3a": 0,
        "min_best_rmsd_to_exp": pd.NA,
        "max_norm_o3a_to_exp": pd.NA,
        "valid": False,
        "rmsd_valid": False,
        "norm_o3a_valid": False if o3a_enabled else pd.NA,
        "status": status,
        "rmsd_status": status,
        "norm_o3a_status": "not_evaluated" if not o3a_enabled else status,
    }

    return add_empty_topk_fields(row, top_k_values, o3a_enabled=o3a_enabled)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    top_k_values = sorted(set(args.top_k))

    if any(k <= 0 for k in top_k_values):
        raise ValueError("All top-k values must be positive integers.")

    if args.bedroc_alpha <= 0:
        raise ValueError("--bedroc-alpha must be greater than zero.")

    if args.ndcg_k <= 0:
        raise ValueError("--ndcg-k must be a positive integer.")

    representatives_top_k_values = sorted(set(args.representatives_top_k))
    if any(k <= 0 for k in representatives_top_k_values):
        raise ValueError("All --representatives-top-k values must be positive integers.")

    if args.norm_o3a_threshold is not None and not (0 <= args.norm_o3a_threshold <= 1.5):
        raise ValueError(
            "--norm-o3a-threshold is expected to be a normalized score, usually 0-1."
        )

    series_col = args.series_col if args.series_col else None
    o3a_enabled = args.norm_o3a_threshold is not None

    csv_files = sorted(
        p for p in args.input_dir.glob(args.pattern) if "_noexp" in p.stem
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No files matching pattern {args.pattern!r} found in {args.input_dir}"
        )

    rows = []
    representative_rows = []
    series_col_found_anywhere = False

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            rows.append(
                make_error_row(
                    file_name=csv_path.name,
                    bag_id=csv_path.stem,
                    status=f"skipped_read_error: {exc}",
                    top_k_values=top_k_values,
                    n_rows_raw=pd.NA,
                    o3a_enabled=o3a_enabled,
                )
            )
            continue

        if series_col is not None and series_col in df.columns:
            series_col_found_anywhere = True

        if args.bag_id_col is not None and args.bag_id_col not in df.columns:
            rows.append(
                make_error_row(
                    file_name=csv_path.name,
                    bag_id=csv_path.stem,
                    status=f"skipped_missing_grouping_column: {args.bag_id_col}",
                    top_k_values=top_k_values,
                    n_rows_raw=len(df),
                    o3a_enabled=o3a_enabled,
                )
            )
            continue

        active_series_col = (
            series_col if series_col is not None and series_col in df.columns else None
        )

        if args.bag_id_col is None:
            rows.append(
                evaluate_bag(
                    df,
                    file_name=csv_path.name,
                    bag_id=csv_path.stem,
                    top_k_values=top_k_values,
                    bedroc_alpha=args.bedroc_alpha,
                    ndcg_k=args.ndcg_k,
                    rmsd_threshold=args.rmsd_threshold,
                    norm_o3a_threshold=args.norm_o3a_threshold,
                    attention_col=args.attention_col,
                    rmsd_col=args.rmsd_col,
                    min_rmsd_col=args.min_rmsd_col,
                    o3a_col=args.o3a_col,
                    max_o3a_col=args.max_o3a_col,
                    series_col=active_series_col,
                    descending_attention=args.descending_attention,
                )
            )
            if args.make_representatives_log:
                representative_rows.extend(
                    make_representative_rows(
                        df,
                        file_name=csv_path.name,
                        bag_id=csv_path.stem,
                        representative_top_k_values=representatives_top_k_values,
                        rmsd_threshold=args.rmsd_threshold,
                        attention_col=args.attention_col,
                        rmsd_col=args.rmsd_col,
                        min_rmsd_col=args.min_rmsd_col,
                        instance_id_col=args.instance_id_col,
                        series_col=active_series_col,
                        descending_attention=args.descending_attention,
                    )
                )
        else:
            for bag_id, bag_df in df.groupby(args.bag_id_col, dropna=False):
                rows.append(
                    evaluate_bag(
                        bag_df,
                        file_name=csv_path.name,
                        bag_id=str(bag_id),
                        top_k_values=top_k_values,
                        bedroc_alpha=args.bedroc_alpha,
                        ndcg_k=args.ndcg_k,
                        rmsd_threshold=args.rmsd_threshold,
                        norm_o3a_threshold=args.norm_o3a_threshold,
                        attention_col=args.attention_col,
                        rmsd_col=args.rmsd_col,
                        min_rmsd_col=args.min_rmsd_col,
                        o3a_col=args.o3a_col,
                        max_o3a_col=args.max_o3a_col,
                        series_col=active_series_col,
                        descending_attention=args.descending_attention,
                    )
                )

    per_bag = pd.DataFrame(rows)
    representatives = pd.DataFrame(representative_rows)

    summary = make_global_summary(
        per_bag=per_bag,
        top_k_values=top_k_values,
        rmsd_threshold=args.rmsd_threshold,
        norm_o3a_threshold=args.norm_o3a_threshold,
        ttest_alternative=args.ttest_alternative,
    )

    ranking_summary = make_ranking_metrics_summary(
        per_bag=per_bag,
        rmsd_threshold=args.rmsd_threshold,
        norm_o3a_threshold=args.norm_o3a_threshold,
        bedroc_alpha=args.bedroc_alpha,
        ndcg_k=args.ndcg_k,
    )

    series_reporting_enabled = bool(series_col_found_anywhere)

    if series_reporting_enabled:
        series_overview, series_summary = make_series_reports(
            per_bag=per_bag,
            top_k_values=top_k_values,
            rmsd_threshold=args.rmsd_threshold,
            norm_o3a_threshold=args.norm_o3a_threshold,
            ttest_alternative=args.ttest_alternative,
        )
        series_ranking_summary = make_series_ranking_metrics_summary(
            per_bag=per_bag,
            rmsd_threshold=args.rmsd_threshold,
            norm_o3a_threshold=args.norm_o3a_threshold,
            bedroc_alpha=args.bedroc_alpha,
            ndcg_k=args.ndcg_k,
        )
    else:
        series_overview, series_summary = pd.DataFrame(), pd.DataFrame()
        series_ranking_summary = pd.DataFrame()

    per_bag_path = args.output_prefix.with_name(args.output_prefix.name + "_per_bag.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    ranking_summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_ranking_metrics_summary.csv"
    )
    series_ranking_summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_series_ranking_metrics_summary.csv"
    )
    series_overview_path = args.output_prefix.with_name(
        args.output_prefix.name + "_series_overview.csv"
    )
    series_summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_series_summary.csv"
    )
    representatives_path = args.output_prefix.with_name(
        args.output_prefix.name + "_series_representatives_top_attended.csv"
    )
    log_path = args.output_prefix.with_name(args.output_prefix.name + "_log.txt")

    per_bag_path.parent.mkdir(parents=True, exist_ok=True)

    per_bag.to_csv(per_bag_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranking_summary.to_csv(ranking_summary_path, index=False)
    if args.make_representatives_log:
        representatives.to_csv(representatives_path, index=False)

    if series_reporting_enabled:
        series_overview.to_csv(series_overview_path, index=False)
        series_summary.to_csv(series_summary_path, index=False)
        series_ranking_summary.to_csv(series_ranking_summary_path, index=False)

    barplot_paths: list[Path] = []
    if args.make_barplot:
        barplot_paths = make_topk_barplot(
            summary=summary,
            output_prefix=args.output_prefix,
            plot_top_k_values=sorted(set(args.plot_top_k)),
            alpha=args.ttest_alpha,
            significance_correction=args.plot_significance_correction,
            formats=args.plot_formats,
            dpi=args.plot_dpi,
        )

    log_lines = make_log_lines(
        summary=summary,
        series_overview=series_overview,
        series_summary=series_summary,
        n_total_bags=len(per_bag),
        rmsd_threshold=args.rmsd_threshold,
        norm_o3a_threshold=args.norm_o3a_threshold,
        attention_col=args.attention_col,
        rmsd_col=args.rmsd_col,
        min_rmsd_col=args.min_rmsd_col,
        o3a_col=args.o3a_col,
        max_o3a_col=args.max_o3a_col,
        series_col=series_col,
        series_reporting_enabled=series_reporting_enabled,
        ttest_alternative=args.ttest_alternative,
        ttest_alpha=args.ttest_alpha,
    )

    log_lines.extend(
        make_ranking_metrics_log_lines(
            ranking_summary=ranking_summary,
            series_ranking_summary=series_ranking_summary,
        )
    )

    if args.make_representatives_log:
        log_lines.extend(
            make_representatives_log_lines(
                representatives=representatives,
                rmsd_threshold=args.rmsd_threshold,
            )
        )

    log_text = "\n".join(log_lines)
    log_path.write_text(log_text + "\n", encoding="utf-8")

    print("\n" + log_text)
    print(f"\nPer-bag report saved to: {per_bag_path}")
    print(f"Summary report saved to: {summary_path}")
    print(f"Ranking metrics summary saved to: {ranking_summary_path}")
    if args.make_representatives_log:
        print(f"Series representatives report saved to: {representatives_path}")

    if series_reporting_enabled:
        print(f"Series overview saved to: {series_overview_path}")
        print(f"Series summary saved to: {series_summary_path}")
        print(f"Series ranking metrics summary saved to: {series_ranking_summary_path}")
    else:
        print("Series reports were not generated because the series column was not found.")

    if barplot_paths:
        for path in barplot_paths:
            print(f"Barplot saved to: {path}")
    elif args.make_barplot:
        print("Barplot was requested but was not generated.")
    else:
        print("Barplot generation disabled.")

    print(f"Calculation log saved to: {log_path}")


if __name__ == "__main__":
    main()