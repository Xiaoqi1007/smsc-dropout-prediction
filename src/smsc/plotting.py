"""Publication performance and risk figures from saved predictions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smsc.metrics import evaluate, interpolate_roc_pr

BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#555555"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def save(fig, output, name):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def band(ax, x, mean, sd, color):
    ax.fill_between(
        x,
        np.clip(mean - sd, 0, 1),
        np.clip(mean + sd, 0, 1),
        color=color,
        alpha=0.16,
        linewidth=0,
    )


def spread(values):
    return (
        np.std(values, axis=0, ddof=1)
        if len(values) > 1
        else np.full(np.shape(values)[1:], np.nan)
    )


def performance_data(test, window, calibration_strategy="equal_width", n_bins=10):
    """Equal-seed summaries; missing/single-class subsets remain explicitly undefined."""

    def score_rows(rows):
        result = evaluate(rows.y_true, rows.probability)
        if calibration_strategy == "equal_session_count" and len(rows):
            order = [
                c
                for c in ["probability", "probability_raw", "id", "session"]
                if c in rows
            ]
            ordered = rows.sort_values(order, kind="stable")
            parts = np.array_split(np.arange(len(ordered)), 10)
            result["ece_10"] = sum(
                len(ix)
                / len(ordered)
                * abs(
                    ordered.iloc[ix].probability.mean() - ordered.iloc[ix].y_true.mean()
                )
                for ix in parts
                if len(ix)
            )
        return result

    grid, edges = np.linspace(0, 1, 201), np.linspace(0, 1, 31)
    scores, curves, histograms, bins, cumulative = [], [], [], [], []
    for seed, group in test.groupby("seed"):
        y, p = group.y_true.to_numpy(), group.probability.to_numpy()
        scores.append(dict(seed=seed, **score_rows(group)))
        if group.y_true.nunique() == 2:
            curves.append(interpolate_roc_pr(y, p, grid))
        histograms.append(np.histogram(p, edges)[0] / len(p))
        if calibration_strategy == "equal_session_count":
            order = [
                c
                for c in ["probability", "probability_raw", "id", "session"]
                if c in group
            ]
            parts = np.array_split(
                group.sort_values(order, kind="stable").index.to_numpy(), n_bins
            )
        elif calibration_strategy == "equal_width":
            index = np.minimum((p * n_bins).astype(int), n_bins - 1)
            parts = [group.index[index == b] for b in range(n_bins)]
        else:
            raise ValueError("Unknown calibration binning strategy")
        for b, indexes in enumerate(parts):
            part = group.loc[indexes]
            bins.append(
                dict(
                    seed=seed,
                    bin=b,
                    records=len(part),
                    patients=part.id.nunique(),
                    predicted=part.probability.mean(),
                    observed=part.y_true.mean(),
                )
            )
        for session in range(window[0], window[1] + 1):
            part = group[group.session.between(window[0], session)]
            cumulative.append(dict(seed=seed, session=session, **score_rows(part)))
    bins, cumulative = pd.DataFrame(bins), pd.DataFrame(cumulative)
    summary = cumulative.groupby("session")[["auroc", "auprc", "prevalence"]].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    bin_summary = bins.groupby("bin", as_index=False).agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        std=("observed", "std"),
        n_seeds=("observed", "count"),
    )
    return dict(
        scores=pd.DataFrame(scores),
        curves=np.asarray(curves),
        grid=grid,
        edges=edges,
        histogram=np.mean(histograms, axis=0),
        bin_rows=bins,
        bins=bin_summary,
        cumulative=cumulative,
        summary=summary.reset_index(),
    )


def best_f1(validation):
    """Session-specific validation F1; highest threshold breaks exact ties."""
    if not validation.split.eq("validation").all():
        raise ValueError("Threshold fitting requires validation rows only")
    if validation.y_true.nunique() != 2:
        return np.nan
    y, probability = validation.y_true.to_numpy(), validation.probability.to_numpy()
    thresholds = np.unique(np.r_[0.0, probability, np.nextafter(1.0, 2.0)])
    scores = []
    for threshold in thresholds:
        flag = probability >= threshold
        tp = np.sum((y == 1) & flag)
        fp = np.sum((y == 0) & flag)
        fn = np.sum((y == 1) & ~flag)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    chosen = max(range(len(thresholds)), key=lambda i: (scores[i], thresholds[i]))
    return float(thresholds[chosen])


def risk_data(
    full, window, seed=None, dropout_id=None, completer_id=None, endpoints=None
):
    test = full[full.split.eq("test")]
    displayed = test[test.session.between(*window)]
    candidates = [int(seed)] if seed is not None else sorted(displayed.seed.unique())
    chosen = None
    for candidate in candidates:
        part = displayed[displayed.seed.eq(candidate)]
        ids = []
        for outcome, requested in [(1, dropout_id), (0, completer_id)]:
            available = sorted(part.loc[part.y_true.eq(outcome), "id"].unique())
            ids.append(
                str(requested)
                if requested is not None and str(requested) in available
                else (available[0] if requested is None and available else None)
            )
        if all(pid is not None for pid in ids):
            chosen = int(candidate), ids
            break
    if chosen is None:
        raise ValueError(
            "No eligible test examples: check seed, exact IDs, outcome classes and display window"
        )
    seed, ids = chosen
    within = displayed.groupby(["seed", "session", "y_true"], as_index=False).agg(
        risk=("probability", "mean"), patients=("id", "nunique")
    )
    group = within.groupby(["session", "y_true"], as_index=False).agg(
        risk_mean=("risk", "mean"), risk_std=("risk", "std"), n_seeds=("risk", "count")
    )
    validation = full[full.seed.eq(seed) & full.split.eq("validation")]
    cuts = []
    for session in range(window[0], window[1] + 1):
        part = validation[validation.session.eq(session)]
        threshold = best_f1(part)
        cuts.append(
            dict(
                seed=seed,
                session=session,
                threshold=threshold,
                patients=part.id.nunique(),
                positives=int(part.y_true.sum()),
                status=(
                    "available" if np.isfinite(threshold) else "missing_or_single_class"
                ),
            )
        )
    cuts = pd.DataFrame(cuts)
    examples = displayed[displayed.seed.eq(seed) & displayed.id.isin(ids)].merge(
        cuts[["session", "threshold"]], on="session", how="left", validate="many_to_one"
    )
    examples["crossing"] = (
        examples.probability.ge(examples.threshold) & examples.threshold.notna()
    )
    endpoint = None
    if endpoints is not None:
        endpoint = pd.read_csv(endpoints, dtype={"id": str, "record_id": str})
        if "id" not in endpoint:
            endpoint = endpoint.rename(columns={"record_id": "id"})
        if not {"id", "last_session", "ever_dropout"}.issubset(endpoint):
            raise ValueError(
                "Endpoint table requires id or record_id, last_session and ever_dropout"
            )
        endpoint["id"] = endpoint.id.astype(str)
        if endpoint.id.duplicated().any() or not {
            "id",
            "last_session",
            "ever_dropout",
        }.issubset(endpoint):
            raise ValueError(
                "Endpoint table requires unique string IDs, last_session and ever_dropout"
            )
        endpoint = endpoint.set_index("id")
    audit = []
    for outcome, pid in zip([1, 0], ids):
        rows = test[test.seed.eq(seed) & test.id.eq(pid)]
        last = int(rows.session.max())
        kind = "Last observed in run"
        if endpoint is not None:
            if (
                pid not in endpoint.index
                or endpoint.loc[pid, "ever_dropout"] != outcome
            ):
                raise ValueError(f"Endpoint outcome mismatch for {pid}")
            value = float(endpoint.loc[pid, "last_session"])
            if not np.isfinite(value) or value != int(value) or value < last:
                raise ValueError(f"Invalid attendance endpoint for {pid}")
            last, kind = int(value), "Last attended"
        row = examples[examples.id.eq(pid)]
        flags = row.loc[row.crossing, "session"]
        audit.append(
            dict(
                id=pid,
                outcome=outcome,
                seed=seed,
                last_session=last,
                endpoint_label=kind,
                first_crossing=int(flags.min()) if len(flags) else None,
                n_crossings=len(flags),
            )
        )
    return group, within, cuts, examples, audit


def draw_risk(
    full,
    output,
    window,
    title,
    seed=None,
    dropout_id=None,
    completer_id=None,
    endpoints=None,
):
    group, within, cuts, examples, audit = risk_data(
        full, window, seed, dropout_id, completer_id, endpoints
    )
    sessions = np.arange(window[0], window[1] + 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.80, wspace=0.13)
        for outcome, color in [(0, BLUE), (1, ORANGE)]:
            g = group[group.y_true.eq(outcome)].set_index("session").reindex(sessions)
            mean, sd = g.risk_mean.to_numpy(), g.risk_std.to_numpy()
            axes[0].plot(sessions, mean, color=color, lw=2.6)
            band(axes[0], sessions, mean, sd, color)
        axes[0].set_title("Mean calibrated risk", pad=12)
        axes[0].text(
            0.04,
            0.96,
            f"Across {full.seed.nunique()} test splits\nShading: ±1 SD across splits",
            transform=axes[0].transAxes,
            va="top",
            fontsize=9.5,
            color=GRAY,
        )
        for ax, info, color, label in zip(
            axes[1:], audit, [ORANGE, BLUE], ["Dropout", "Completer"]
        ):
            g = (
                examples[examples.id.eq(info["id"])]
                .set_index("session")
                .reindex(sessions)
            )
            ax.plot(
                sessions, g.probability, "-o", color=color, lw=2.3, ms=4.5, zorder=4
            )
            ax.plot(
                cuts.session,
                cuts.threshold,
                "s--",
                color=GRAY,
                lw=1.4,
                ms=3.5,
                zorder=3,
            )

            for row in cuts.iloc[:: max(1, int(np.ceil(len(cuts) / 10)))].itertuples():
                if np.isfinite(row.threshold):
                    ax.annotate(
                        f"{row.threshold:.3f}",
                        (row.session, row.threshold),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.8,
                        color=GRAY,
                    )
            last = info["last_session"]
            if last < window[1]:
                ax.axvline(last, color="#888888", ls=":", lw=1.4)
                ax.axvspan(last + 0.03, window[1] + 0.25, color="#eeeeee", zorder=0)
                ax.text(
                    (max(window[0], last) + window[1]) / 2,
                    0.55,
                    f"No observations\nafter Session {last}",
                    ha="center",
                    fontsize=8.5,
                    color="#666666",
                )
            description = f'{info["endpoint_label"]}: Session {last}'
            ax.text(
                0.04,
                0.96,
                description,
                transform=ax.transAxes,
                va="top",
                fontsize=8.6,
                color=GRAY,
            )
            ax.set_title(f"{label} example", pad=12)
        for label, ax in zip("ABC", axes):
            ax.set(
                xlim=(window[0] - 0.2, window[1] + 0.2),
                ylim=(0, 1),
                xticks=sessions[:: max(1, int(np.ceil(len(sessions) / 12)))],
                xlabel="Session index",
            )
            ax.grid(axis="y", color="#eeeeee", lw=0.8)
            ax.text(
                -0.075,
                1.055,
                label,
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=18,
            )
        axes[0].set_ylabel("Dropout probability")
        handles = [
            Line2D([], [], color=BLUE, lw=2.6, label="Completed"),
            Line2D([], [], color=ORANGE, lw=2.6, label="Dropout"),
            Line2D(
                [],
                [],
                color=GRAY,
                ls="--",
                marker="s",
                ms=3.5,
                lw=1.4,
                label="Validation F1 threshold (B, C)",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.52, 0.10),
            ncol=3,
            frameon=False,
            fontsize=10,
            handlelength=2.5,
            columnspacing=1.7,
        )
        save(fig, output, "risk_ABC")
    for name, table in [
        ("group_risk_summary", group),
        ("group_risk_by_seed", within),
        ("validation_F1_thresholds", cuts),
        ("individual_examples", examples),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    return audit
