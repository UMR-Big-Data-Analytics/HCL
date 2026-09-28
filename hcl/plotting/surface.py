import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa

from hcl.definitions import ROOT_DIR
from hcl.plotting.consts import (
    PT_PER_INCH,
    figure_size,
    figure_width_pt,
    save_figure,
    setup_publication_style,
)


def _square_panel_figsize(
    rows: int,
    cols: int,
    *,
    columns: int,
    decoration_in: float = 0.0,
    sidebar_in: float = 0.0,
    scale: float = 1.0,
):
    """Starting figure size for a rows x cols grid of square panels spanning ``columns`` columns.

    ``sidebar_in`` is the horizontal space taken by the y axis labels and ``decoration_in``
    the vertical space taken by titles, x labels and the colorbar. The height only has to be
    generous enough for the panels to be limited by the figure width; the surplus is removed
    afterwards by :func:`_fit_square_grid_height`.
    """
    width_in = figure_width_pt(columns) / PT_PER_INCH
    panel_in = (width_in - sidebar_in) / cols
    aspect = (rows * panel_in + decoration_in) / width_in
    return figure_size(columns=columns, aspect=aspect, scale=scale)


def _fit_square_grid_height(fig, axes, *, tol_in: float = 0.001, iterations: int = 14) -> None:
    """Trim the figure height to the smallest value that keeps the panels at full size.

    The figure is created with a deliberately generous height so the square panels are
    limited by the (fixed) figure width. The surplus height would otherwise be spent as
    whitespace between the rows, so it is removed by a bisection on the figure height that
    also makes sure nothing (titles, labels, colorbar) is pushed outside the canvas.
    """
    grid = np.atleast_2d(np.asarray(axes, dtype=object))

    def measure() -> tuple[float, bool]:
        fig.canvas.draw()
        panel_in = min(ax.get_window_extent().height for ax in grid.flat) / fig.dpi
        bbox = fig.get_tightbbox()
        inside = bbox.y0 >= -tol_in and bbox.y1 <= fig.get_figheight() + tol_in
        return panel_in, inside

    target, _ = measure()
    lo, hi = 0.0, fig.get_figheight()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fig.set_figheight(mid)
        panel_in, inside = measure()
        if inside and panel_in >= target - tol_in:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol_in:
            break
    fig.set_figheight(hi)
    fig.canvas.draw()


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)

    def lr(N_X: int, R_X: float, N_Y: int, R_Y: float, theta: float, gamma: float) -> float:
        cos_theta = np.cos(np.deg2rad(theta))
        N_XY = N_X + N_Y
        R_XY = np.sqrt(R_X**2 + R_Y**2 + 2 * R_X * R_Y * cos_theta)

        term_XY = N_XY * np.log(1 - R_XY / N_XY + gamma)
        term_X = N_X * np.log(1 - R_X / N_X + gamma)
        term_Y = N_Y * np.log(1 - R_Y / N_Y + gamma)

        return term_XY - term_X - term_Y

    # Fixed properties for A and C
    N_A, R_A = 10, 8.0  # R_bar = 0.8
    N_C, R_C = 10, 8.0  # R_bar = 0.8
    N_B = 10

    gammas = [0.001, 0.01, 0.1, 1]
    R_bars_B = [0.5, 0.8, 0.95]

    theta_AB_vals = np.linspace(0.0, 180.0, 100)
    theta_AC_vals = np.linspace(0.0, 180.0, 100)
    X, Y = np.meshgrid(theta_AB_vals, theta_AC_vals)

    fig, axes = plt.subplots(
        len(R_bars_B),
        len(gammas),
        figsize=_square_panel_figsize(
            len(R_bars_B), len(gammas), columns=2, decoration_in=1.6, sidebar_in=0.55, scale=0.8
        ),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for i, R_bar_B in enumerate(R_bars_B):
        R_B = N_B * R_bar_B
        for j, gamma in enumerate(gammas):
            Z = np.zeros_like(X)
            for r in range(X.shape[0]):
                for c in range(X.shape[1]):
                    cost_AB = lr(N_A, R_A, N_B, R_B, X[r, c], gamma)
                    cost_AC = lr(N_A, R_A, N_C, R_C, Y[r, c], gamma)
                    Z[r, c] = cost_AB - cost_AC

            ax = axes[i, j]
            # Normalize cmap nicely. We care about the sign.
            vmax = np.max(np.abs(Z))
            if vmax == 0:
                vmax = 1  # avoid division by zero

            # Plot heatmap
            im = ax.pcolormesh(
                X,
                Y,
                Z,
                cmap="RdBu",
                vmin=-vmax,
                vmax=vmax,
                shading="nearest",
                antialiased=False,
                rasterized=True,
            )
            # Add a contour at 0
            ax.contour(X, Y, Z, levels=[0], colors="k", linestyles="--")

            if i == 0:
                ax.set_title(f"$\\gamma = {gamma}$")
            if j == 0:
                ax.set_ylabel(f"$\\bar{{R}}_B = {R_bar_B}$\n$\\theta_{{AC}}$")
            if i == len(R_bars_B) - 1:
                ax.set_xlabel("$\\theta_{AB}$")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_yticks([0, 45, 90, 135, 180])
            ax.set_box_aspect(1)

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        location="bottom",
        fraction=0.04,
        pad=0.06,
        aspect=45,
    )
    cbar.set_label("$\\Delta=\\mathrm{LR}(A,B) - \\mathrm{LR}(A,C)$")
    # _fit_square_grid_height(fig, axes)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "LR_surface_N_B_10.pdf")
    plt.close(fig)

    # Fixed properties for A and C
    N_A, R_A = 10, 8.0  # R_bar = 0.8
    N_C, R_C = 10, 8.0  # R_bar = 0.8

    N_Bs = [5, 10, 15]
    gamma = 0.01

    R_bars_B = [0.8]

    theta_AB_vals = np.linspace(0.0, 180.0, 100)
    theta_AC_vals = np.linspace(0.0, 180.0, 100)
    X, Y = np.meshgrid(theta_AB_vals, theta_AC_vals)

    fig, axes = plt.subplots(
        len(R_bars_B),
        len(N_Bs),
        figsize=_square_panel_figsize(
            len(R_bars_B), len(N_Bs), columns=1, decoration_in=1.6, sidebar_in=0.45, scale=0.8
        ),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for i, R_bar_B in enumerate(R_bars_B):
        for j, N_B in enumerate(N_Bs):
            R_B = N_B * R_bar_B
            Z = np.zeros_like(X)
            for r in range(X.shape[0]):
                for c in range(X.shape[1]):
                    cost_AB = lr(N_A, R_A, N_B, R_B, X[r, c], gamma)
                    cost_AC = lr(N_A, R_A, N_C, R_C, Y[r, c], gamma)
                    Z[r, c] = cost_AB - cost_AC

            if len(R_bars_B) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            # Normalize cmap nicely. We care about the sign.
            vmax = np.max(np.abs(Z))
            if vmax == 0:
                vmax = 1  # avoid division by zero

            # Plot heatmap
            im = ax.pcolormesh(
                X,
                Y,
                Z,
                cmap="RdBu",
                vmin=-vmax,
                vmax=vmax,
                shading="nearest",
                antialiased=False,
                rasterized=True,
            )
            # Add a contour at 0
            ax.contour(X, Y, Z, levels=[0], colors="k", linestyles="--")

            if i == 0:
                ax.set_title(f"$N_B = {N_B}$")
            if j == 0:
                ax.set_ylabel("$\\theta_{AC}$")
            ax.set_xticks([0, 90, 180])
            ax.set_yticks([0, 90, 180])
            ax.set_box_aspect(1)

    # A single row of panels: the constant parameters go into the supertitle and the shared
    # x label is drawn once under the middle panel instead of below every panel.
    # fig.suptitle(f"$\\bar{{R}}_B = {R_bars_B[0]}$, $\\gamma = {gamma}$")
    axes[len(N_Bs) // 2].set_xlabel("$\\theta_{AB}$")
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        location="bottom",
        fraction=0.08,
        pad=0.05,
        aspect=30,
    )
    cbar.set_label("$\\Delta=\\mathrm{LR}(A,B) - \\mathrm{LR}(A,C)$")
    _fit_square_grid_height(fig, axes)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "LR_surface_N_B_vary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
