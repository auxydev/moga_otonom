"""Time-series plots of simulation results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_localization(result, path: str, title: str | None = None,
                      shade: list[tuple[float, float, str]] | None = None):
    log = result.log
    t = np.asarray(log["loc_t"])
    e = np.asarray(log["loc_err"])
    sd = np.asarray(log["loc_std"])
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), dpi=110, sharex=True)
    ax[0].plot(t, e, label="konum hatası |x̂ - x| [m]", color="#1f6fd1")
    ax[0].plot(t, 3 * sd, label="3σ (EKF kovaryansı)", color="#e8833a", ls="--")
    ax[0].set_ylabel("m")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[1].plot(log["t"], np.asarray(log["v"]) * 3.6, color="#2a9d4b")
    ax[1].set_ylabel("hız [km/h]")
    ax[1].set_xlabel("zaman [s]")
    ax[1].grid(alpha=0.3)
    for a, b, lbl in shade or []:
        for x in ax:
            x.axvspan(a, b, color="#999999", alpha=0.2)
        ax[0].text(0.5 * (a + b), ax[0].get_ylim()[1] * 0.9, lbl, ha="center")
    fig.suptitle(title or result.scenario)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
