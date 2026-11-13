"""Build every lecture 10 illustration with Python/matplotlib.

Run from any working directory: python generate_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE = "#003D7C"
RED = "#E52E5A"
GREEN = "#2A7F62"
GRAY = "#6B7280"
PALE = "#B8CDE0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=170)
    plt.close(fig)


def contour(ax, A, lim, levels, title=""):
    xx, yy = np.meshgrid(np.linspace(*lim[0], 350), np.linspace(*lim[1], 350))
    zz = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2)
    ax.contour(xx, yy, zz, levels=levels, colors=PALE, linewidths=1.2)
    ax.scatter([0], [0], color=GREEN, edgecolor="white", linewidth=1, s=55, zorder=7)
    ax.set(xlim=lim[0], ylim=lim[1], aspect="equal", title=title)
    ax.grid(alpha=0.12)


# 1. The metric P changes what a unit displacement means.
t = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2.5, label=r"$\|h\|_2=1$")
ax.plot(np.cos(t) / 3, np.sin(t), color=RED, lw=2.5, label=r"$h^\top P h=1$, $P=\mathrm{diag}(9,1)$")
ax.arrow(0, 0, 0.82, 0, color=BLUE, width=0.012, head_width=0.08, length_includes_head=True)
ax.arrow(0, 0, 0, 0.82, color=GREEN, width=0.012, head_width=0.08, length_includes_head=True)
ax.text(0.61, 0.12, "первое направление", color=BLUE)
ax.text(0.10, 0.63, "второе направление", color=GREEN)
ax.set(xlim=(-1.25, 1.6), ylim=(-1.2, 1.25), aspect="equal", xlabel=r"$h_1$", ylabel=r"$h_2$")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=9)
ax.grid(alpha=0.15)
save(fig, "metric_balls")


# 2. Exact diagonal rescaling makes the model circular in new coordinates.
fig, axs = plt.subplots(1, 2, figsize=(8.7, 3.4))
contour(axs[0], np.diag([25., 1.]), ((-0.9, 0.9), (-2.2, 2.2)), [0.2, 0.5, 1, 2], "Исходные координаты $x$")
contour(axs[1], np.eye(2), ((-2.2, 2.2), (-2.2, 2.2)), [0.2, 0.5, 1, 2], "Новые координаты $y=(5x_1,x_2)$")
axs[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
axs[1].set(xlabel=r"$y_1$", ylabel=r"$y_2$")
save(fig, "ellipse_scaling")


# Shared correlated SPD example for trajectories and spectral comparison.
A = np.array([[100., 9.], [9., 1.]])
P = np.diag(np.diag(A))
Pinvhalf = np.diag(1 / np.sqrt(np.diag(P)))
B = Pinvhalf @ A @ Pinvhalf
eigA, eigB = np.linalg.eigvalsh(A), np.linalg.eigvalsh(B)
x0 = np.array([1., 5.])


def iterate(M, preconditioner, alpha, start, n):
    points = [start.copy()]
    x = start.copy()
    for _ in range(n):
        x = x - alpha * np.linalg.solve(preconditioner, M @ x)
        points.append(x.copy())
    return np.array(points)


gd = iterate(A, np.eye(2), 1 / eigA[-1], x0, 70)
pgd = iterate(A, P, 1 / eigB[-1], x0, 70)


# 3. Five explicit steps on one objective, with both paths and function values.
Apath = np.array([[10., 2.], [2., 1.]])
Ppath = np.diag(np.diag(Apath))
xpath0 = np.array([1., 3.])
alpha_gd = 1 / np.linalg.eigvalsh(Apath)[-1]
alpha_p = 0.30
gd5 = iterate(Apath, np.eye(2), alpha_gd, xpath0, 5)
pgd5 = iterate(Apath, Ppath, alpha_p, xpath0, 5)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.45), gridspec_kw={"width_ratios": [1.15, 1.]})
ax = axs[0]
contour(ax, Apath, ((-.7, 1.15), (-.35, 3.3)), [.1, .4, 1, 2, 4, 8, 15])
ax.set_aspect("auto")
ax.plot(gd5[:, 0], gd5[:, 1], "o-", color=RED, lw=2.2, ms=5, label="обычный GD")
ax.plot(pgd5[:, 0], pgd5[:, 1], "o-", color=BLUE, lw=2.2, ms=5, label="GD с $P$")
ax.scatter(*xpath0, color=GRAY, edgecolor="white", linewidth=1, s=75, zorder=8)
ax.annotate("$x_0$", xpath0, xytext=(-8, -17), textcoords="offset points", color=GRAY)
ax.annotate("$x^*=0$", (0, 0), xytext=(8, 3), textcoords="offset points", color=GREEN)
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Путь: ровно 5 отрезков")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=8)
ax = axs[1]
for path, color, label in [(gd5, RED, "обычный GD"), (pgd5, BLUE, "GD с $P$")]:
    values = 0.5 * np.einsum("ki,ij,kj->k", path, Apath, path)
    ax.plot(range(6), values, "o-", color=color, lw=2.3, ms=5, label=label)
ax.set(xlim=(-.1, 5.1), ylim=(0, 16.5), xticks=range(6),
       xlabel="номер шага $k$", ylabel=r"$f(x_k)$", title="Значение функции после каждого шага")
ax.grid(alpha=.18)
save(fig, "path_comparison")


# 4. The spectrum changes substantially under Jacobi scaling.
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
for ax, vals, title, kappa in [
    (axs[0], eigA, "Исходная матрица $A$", eigA[-1] / eigA[0]),
    (axs[1], eigB, r"$B=P^{-1/2}AP^{-1/2}$", eigB[-1] / eigB[0]),
]:
    bars = ax.bar([0, 1], vals, color=[GREEN, BLUE], width=0.58)
    ax.set(yscale="log", xticks=[0, 1], xticklabels=[r"$\lambda_{\min}$", r"$\lambda_{\max}$"], title=title)
    ax.text(0.5, 0.92, rf"$\kappa\approx {kappa:.0f}$", transform=ax.transAxes, ha="center", color=RED, fontsize=12)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12, f"{v:.2g}", ha="center", fontsize=9)
axs[0].set_ylabel("собственное значение (логарифмическая шкала)")
axs[0].set_ylim(0.07, 230)
save(fig, "eigen_before_after")


# 5. Worst-case spectral contraction as a function of step length.
alpha = np.linspace(0, 1.08, 500)
rho1 = np.abs(1 - eigB[0] * alpha)
rho2 = np.abs(1 - eigB[-1] * alpha)
rho = np.maximum(rho1, rho2)
alpha_star = 2 / eigB.sum()
rho_star = (eigB[-1] - eigB[0]) / eigB.sum()
fig, ax = plt.subplots(figsize=(7.8, 3.35))
ax.plot(alpha, rho1, color=GREEN, lw=1.8, ls="--", label=r"$|1-\alpha\lambda_{\min}|$")
ax.plot(alpha, rho2, color=RED, lw=1.8, ls="--", label=r"$|1-\alpha\lambda_{\max}|$")
ax.plot(alpha, rho, color=BLUE, lw=2.7, label=r"$\rho(\alpha)$")
ax.scatter([alpha_star], [rho_star], color=BLUE, edgecolor="white", linewidth=1.2, s=90, zorder=5)
ax.annotate(r"$\alpha_*=1$, $\rho_*=0.9$", (alpha_star, rho_star), xytext=(-160, -45), textcoords="offset points", color=BLUE, fontsize=10, arrowprops=dict(arrowstyle="->", color=BLUE))
ax.set(xlim=(0, 1.08), ylim=(0, 1.28), xlabel=r"шаг $\alpha$", ylabel="максимальный модуль множителя")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9)
ax.grid(alpha=0.18)
save(fig, "step_spectrum")


# 6. Objective convergence for the same starting point.
def energy(points, M):
    return 0.5 * np.einsum("ki,ij,kj->k", points, M, points)

fig, ax = plt.subplots(figsize=(7.6, 3.4))
k = np.arange(len(gd))
ax.semilogy(k, energy(gd, A), color=RED, lw=2.4, label="обычный GD")
ax.semilogy(k, energy(pgd, A), color=BLUE, lw=2.4, label="диагональное предобусловливание")
ax.set(xlim=(0, 70), xlabel="номер шага", ylabel=r"$f(x_k)-f^*$ (логарифмическая шкала)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.18, which="both")
save(fig, "convergence_comparison")


# 7. A diagonal preconditioner cannot remove a 45-degree rotation.
R = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
Arot = R @ np.diag([1., 100.]) @ R.T
Prot = np.diag(np.diag(Arot))
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"width_ratios": [1.25, 0.8]})
contour(axs[0], Arot, ((-1.3, 1.3), (-1.3, 1.3)), [0.2, 1, 4, 12, 30], "Повернутая квадратичная функция")
axs[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
axs[1].bar([0, 1], [100, 100], color=[RED, BLUE], width=0.55)
axs[1].set(ylim=(0, 122), ylabel=r"число обусловленности $\kappa$", xticks=[0, 1], xticklabels=["до", "после $P=\\mathrm{diag}(A)$"])
axs[1].grid(axis="y", alpha=0.17)
axs[1].set_axisbelow(True)
save(fig, "rotated_failure")


# 8. Feature scaling for a small ridge regression design matrix.
Xbase = np.array([[1., .2], [2., -1.], [-1., 1.5], [.5, 2.]])
X = Xbase @ np.diag([15., 1.])
lam = .1
Aridge = X.T @ X / len(X) + lam * np.eye(2)
Pridge = np.diag(np.diag(Aridge))
Br = np.diag(1 / np.sqrt(np.diag(Pridge))) @ Aridge @ np.diag(1 / np.sqrt(np.diag(Pridge)))
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
contour(axs[0], Aridge, ((-.5, .5), (-2., 2.)), [.2, .5, 1, 2, 5], "Весовые коэффициенты $w$")
axs[0].set_aspect("auto")  # Horizontal and vertical coordinates have different scales.
contour(axs[1], Br, ((-2., 2.), (-2., 2.)), [.2, .5, 1, 2], "Масштабированные координаты $u$")
axs[0].set(xlabel=r"$w_1$", ylabel=r"$w_2$")
axs[1].set(xlabel=r"$u_1$", ylabel=r"$u_2$")
save(fig, "ridge_feature_scaling")

print(f"Generated {len(list(OUT.glob('*.pdf')))} PDF figures")
print("A eigenvalues:", eigA, "condition number:", eigA[-1] / eigA[0])
print("B eigenvalues:", eigB, "condition number:", eigB[-1] / eigB[0])
print("Ridge condition numbers:", np.linalg.cond(Aridge), np.linalg.cond(Br))
