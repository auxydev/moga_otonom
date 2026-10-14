"""Live top-down view of a running MogaSim scenario (interactive window)."""
from __future__ import annotations

import sys
import time


def _interactive_backend(plt) -> bool:
    """Switch pyplot to an interactive backend (the renderer defaults to Agg)."""
    candidates = ["MacOSX"] if sys.platform == "darwin" else []
    candidates += ["TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"]
    for name in candidates:
        try:
            plt.switch_backend(name)
            fig = plt.figure()
            plt.close(fig)
            return True
        except Exception:  # noqa: BLE001 - backend not available on this system
            continue
    plt.switch_backend("Agg")
    return False


def run_live(name: str, driver: str | None = None, seed: int = 0, speed: float = 1.0,
             frame_dt: float = 0.1):
    """Run scenario ``name`` and draw every frame while it runs.

    ``speed``: 1.0 = real time, 2.0 = twice as fast, 0 = as fast as possible.
    Closing the window stops the run."""
    import matplotlib.pyplot as plt

    from moga import scenarios
    from moga.eval.runner import make_driver
    from moga.sim.simulation import Simulation
    from moga.viz.render import Renderer

    interactive = _interactive_backend(plt)
    if not interactive:
        print("No interactive matplotlib backend (install tkinter or PyQt); running without a window.")
    scn = scenarios.get(name, seed=seed)
    drv = make_driver(driver, scn)
    rend = Renderer(scn.map, figsize=(12.0, 6.75))
    t_start = time.time()

    class Stop(Exception):
        pass

    def on_frame(fr: dict):
        rend.draw_frame(fr)
        if not interactive:
            return
        if rend.fig.canvas.manager is not None:
            rend.fig.canvas.manager.set_window_title(f"MogaSim – {name}  t={fr['t']:.1f} s")
        if not plt.fignum_exists(rend.fig.number):
            raise Stop
        wait = fr["t"] / speed - (time.time() - t_start) if speed > 0 else 0.0
        plt.pause(max(wait, 0.001))
        sim.frames.clear()                 # nothing is kept: memory stays flat

    sim = Simulation(scn, drv, frame_dt=frame_dt, on_frame=on_frame)
    if interactive:
        plt.ion()
        rend._setup()
        plt.show(block=False)
    try:
        res = sim.run()
    except (Stop, KeyboardInterrupt):
        print("stopped")
        return None
    print(res.short())
    if interactive and plt.fignum_exists(rend.fig.number):
        print("close the window to exit")
        plt.ioff()
        plt.show()
    return res
