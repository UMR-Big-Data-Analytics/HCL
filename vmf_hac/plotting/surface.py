import os
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from vmf_hac.definitions import ROOT_DIR


def main():
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
            }
        )
    except:  # noqa
        warn("Warning: LaTeX not available. Using default matplotlib text rendering.", stacklevel=2)
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
        len(R_bars_B), len(gammas), figsize=(3 * len(gammas) + 1, 3 * len(R_bars_B)), sharex=True, sharey=True
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
                ax.set_xlabel("$\\theta_{{AB}}$")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_yticks([0, 45, 90, 135, 180])

    cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])  # ty:ignore[no-matching-overload]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    # cbar.set_label('$\\leftarrow$ Merge (A,B)$\\quad$ |$\\quad lr(A,B) - lr(A,C)\\quad$ |$\\quad$ Merge (A,C) $\\rightarrow$', fontsize=12)
    cbar.set_label("$\\Delta=\\mathrm{LR}(A,B) - \\mathrm{LR}(A,C)$", fontsize=12)
    fig.subplots_adjust(bottom=0.05)
    plt.savefig(ROOT_DIR / "results" / "plots" / "LR_surface_N_B_10.pdf", bbox_inches="tight", dpi=300)

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
        len(R_bars_B), len(N_Bs), figsize=(2 * len(N_Bs), 2 * len(R_bars_B) + 0.4), sharex=True, sharey=True
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
                ax.set_ylabel(f"$\\bar{{R}}_B = {R_bar_B}$\n$\\theta_{{AC}}$")
            if i == len(R_bars_B) - 1:
                ax.set_xlabel("$\\theta_{{AB}}$")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_yticks([0, 45, 90, 135, 180])

    cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03])  # ty:ignore[no-matching-overload]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    # cbar.set_label('$\\leftarrow$ Merge (A,B)$\\quad$ |$\\quad lr(A,B) - lr(A,C)\\quad$ |$\\quad$ Merge (A,C) $\\rightarrow$', fontsize=12)
    cbar.set_label("$\\Delta=\\mathrm{LR}(A,B) - \\mathrm{LR}(A,C)$", fontsize=12)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / "results" / "plots" / "LR_surface_N_B_vary.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
