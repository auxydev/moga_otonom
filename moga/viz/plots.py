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


def camera_overlay(image: np.ndarray, camera, lanes, path: str | None = None) -> np.ndarray:
    """RGB camera image with detected lane boundaries projected back."""
    rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if lanes is not None:
        for bd, col in ((lanes.left, (255, 60, 230)), (lanes.right, (60, 255, 120))):
            if bd is None:
                continue
            xs = np.linspace(4.5, max(bd.x_max, 6.0), 300)
            pts = np.column_stack([xs, bd.y(xs), np.zeros_like(xs)])
            u, v, ok = camera.project(pts)
            for uu, vv, o in zip(u, v, ok):
                if not o or not np.isfinite(uu):
                    continue
                ui, vi = int(round(uu)), int(round(vv))
                if 1 <= ui < rgb.shape[1] - 1 and 1 <= vi < rgb.shape[0] - 1:
                    rgb[vi - 1:vi + 2, ui - 1:ui + 2] = col
    if path:
        from PIL import Image
        Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.NEAREST).save(path)
    return rgb
