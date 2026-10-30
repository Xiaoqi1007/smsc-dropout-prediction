"""Figure 2: discrimination, calibration and cumulative metrics."""

import numpy as np
import matplotlib.pyplot as plt
from .plotting import performance_data

BLUE = "#087dcc"
ROSE = "#c42d63"
GRAY = "#686868"


def draw(combined, FIGURES, TABLES):
    def save(fig, name):
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(
                FIGURES / f"{name}.{ext}",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)

    data = {
        m: performance_data(g, (0, 23), "equal_session_count", 10)
        for m, g in combined.loc[combined.split.eq("test")].groupby("model")
    }
    fig = plt.figure(figsize=(13.4, 8.1), layout="constrained")
    grid = fig.add_gridspec(2, 6, hspace=0.12, wspace=0.12)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]
    for m, color, ls, label in [
        ("full_model", BLUE, "-", "Multimodal"),
        ("clinician_baseline", ROSE, "--", "Clinician-rating"),
    ]:
        d = data[m]
        s = d["scores"]
        line_zorder = 4 if m == "full_model" else 3
        for j, metric in [(0, "auroc"), (1, "auprc")]:
            values = d["curves"][:, j]
            mean = values.mean(0)
            sd = values.std(0, ddof=1)
            axes[j].plot(
                d["grid"],
                mean,
                color=color,
                ls=ls,
                lw=1.8,
                zorder=line_zorder,
                label=f"{label} {s[metric].mean():.3f} ± {s[metric].std():.3f}",
            )
            axes[j].fill_between(
                d["grid"],
                np.clip(mean - sd, 0, 1),
                np.clip(mean + sd, 0, 1),
                color=color,
                alpha=0.13,
                lw=0,
            )
        b = d["bins"]
        yerr = np.minimum(b["std"].fillna(0), np.minimum(b.observed, 1 - b.observed))
        axes[2].errorbar(
            b.predicted,
            b.observed,
            yerr=yerr,
            color=color,
            ls=ls,
            marker="o",
            ms=3,
            capsize=2,
            lw=1.2,
            zorder=line_zorder,
            label=f"{label}\nBrier {s.brier.mean():.3f} ± {s.brier.std():.3f}",
        )
        for ax, metric in [(axes[3], "auroc"), (axes[4], "auprc")]:
            cum = d["summary"]
            x = cum.session.to_numpy()
            mean = cum[metric + "_mean"].to_numpy()
            sd = cum[metric + "_std"].to_numpy()
            ax.plot(
                x,
                mean,
                color=color,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=2.5,
                zorder=line_zorder,
                label=f"{label} (0–23): {mean[-1]:.3f} ± {sd[-1]:.3f}",
            )
            ax.fill_between(
                x,
                np.clip(mean - sd, 0, 1),
                np.clip(mean + sd, 0, 1),
                color=color,
                alpha=0.13,
                lw=0,
            )
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[1].axhline(
        data["full_model"]["scores"].prevalence.mean(), color="#999999", ls="--", lw=1
    )
    axes[2].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    hist = data["full_model"]
    axes[2].bar(
        hist["edges"][:-1],
        hist["histogram"],
        width=np.diff(hist["edges"]) * 0.9,
        align="edge",
        color="#dddddd",
        zorder=0,
    )
    axes[3].axhline(0.5, color="#999999", ls="--", lw=1)
    axes[4].plot(
        data["full_model"]["summary"].session,
        data["full_model"]["summary"].prevalence_mean,
        color="#777777",
        ls="--",
        lw=1,
        label="Cumulative dropout prevalence",
    )
    for ax, label, title in zip(
        axes,
        "ABCDE",
        ["AUROC", "AUPRC", "Calibration", "Cumulative AUROC", "Cumulative AUPRC"],
    ):
        ax.set_title(title, fontsize=11)
        ax.text(-0.10, 1.05, label, transform=ax.transAxes, weight="bold", fontsize=15)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=7.5, loc="best")
        ax.grid(axis="y", alpha=0.15)
    for ax in axes[:3]:
        ax.set_xlim(0, 1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[2].set(
        xlabel="Mean predicted probability", ylabel="Observed dropout frequency"
    )
    for ax, metric in [(axes[3], "AUROC"), (axes[4], "AUPRC")]:
        ax.set(
            xlabel="Latest included session (cumulative from session 0)",
            ylabel=metric,
            xlim=(0, 23),
            xticks=range(0, 24, 2),
        )
    save(fig, "Figure_2_performance")
    for m, d in data.items():
        d["bin_rows"].to_csv(TABLES / f"calibration_bins_{m}.csv", index=False)
