"""Generate all figures for lecture 11 with Python and matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE = "#003D7C"
PINK = "#E52E5A"
GREEN = "#2A7F62"
GRAY = "#5F6B76"
PALE = "#B8CDE0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout(pad=0.85)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.08, dpi=190)
    plt.close(fig)


def heavy_ball(A, x0, alpha, beta, steps):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x_next = x - alpha * (A @ x) + beta * (x - x_prev)
        x_prev, x = x, x_next
        path.append(x.copy())
    return np.array(path)


def gd(A, x0, alpha, steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x = x - alpha * (A @ x)
        path.append(x.copy())
    return np.array(path)


def quadratic_contours(ax, A, span=2.4):
    t = np.linspace(-span, span, 400)
    X, Y = np.meshgrid(t, t)
    Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)
    ax.contour(X, Y, Z, levels=[0.2, 0.8, 2, 5, 12, 30, 65, 120],
               colors=PALE, linewidths=1.0)
    ax.scatter([0], [0], s=38, marker="*", c=GREEN, zorder=6)
    ax.set(xlim=(-span, span), ylim=(-span, span), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)


# One rotated quadratic is used throughout the lecture for honest comparison.
theta = np.deg2rad(28)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
A = R @ np.diag([1.0, 100.0]) @ R.T
x0 = R @ np.array([2.0, 0.25])
alpha_gd = 2 / 101
alpha_hb = 4 / (11**2)
beta_hb = (9 / 11)**2

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharex=True, sharey=True)
for ax, path, color, title in [
    (axes[0], gd(A, x0, alpha_gd, 5), BLUE, "Градиентный спуск: 5 шагов"),
    (axes[1], heavy_ball(A, x0, alpha_hb, beta_hb, 5), PINK,
     "Метод тяжёлого шарика: 5 шагов"),
]:
    quadratic_contours(ax, A)
    ax.plot(path[:, 0], path[:, 1], "-o", color=color, lw=1.8, ms=3.7)
    ax.scatter(path[0, 0], path[0, 1], marker="s", s=35, c=color, zorder=7)
    ax.scatter(path[-1, 0], path[-1, 1], marker="o", s=48,
               facecolors="white", edgecolors=color, linewidths=1.6, zorder=7)
    ax.set_title(title)
save(fig, "trajectory_comparison")

I = np.eye(1)
p = heavy_ball(I, [1], 1.0, 0.9, 10)[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2))
axes[0].axhline(0, color=GRAY, lw=0.8)
axes[0].plot(range(len(p)), p, "-o", color=BLUE, ms=4)
axes[0].set(xlabel="номер шага $k$", ylabel="$x_k$", title="Точки пересекают минимум")
axes[1].plot(range(len(p)), 0.5 * p**2, "-o", color=PINK, ms=4)
axes[1].set(xlabel="номер шага $k$", ylabel="$f(x_k)$", title="Значение функции может расти")
for ax in axes:
    ax.grid(alpha=0.18)
save(fig, "nonmonotone_one_dim")

alpha, beta = 0.4, 0.7
path = heavy_ball(np.eye(1), [1], alpha, beta, 14)[:, 0]
vel = np.r_[0.0, np.diff(path)]
fig, ax = plt.subplots(figsize=(5.9, 3.5))
ax.axhline(0, color=GRAY, lw=0.8)
ax.axvline(0, color=GRAY, lw=0.8)
ax.plot(path, vel, "-o", color=BLUE, lw=1.7, ms=3.9)
ax.scatter(path[0], vel[0], marker="s", s=48, color=PINK, zorder=5)
for k in [0, 1, 2, 3, 5, 8, 14]:
    ax.annotate(str(k), (path[k], vel[k]), xytext=(5, 4),
                textcoords="offset points", fontsize=8)
ax.set(xlabel="положение $e_k=x_k-x^*$", ylabel="скорость $v_k=x_k-x_{k-1}$",
       title="Состояние: положение и скорость")
ax.grid(alpha=0.17)
save(fig, "phase_portrait")

fig, ax = plt.subplots(figsize=(6.2, 3.5))
beta_grid = np.linspace(0, 0.995, 300)
ax.fill_betweenx(beta_grid, 0, 2 * (1 + beta_grid), color="#DCEAF6")
ax.plot(2 * (1 + beta_grid), beta_grid, color=BLUE, lw=2,
        label=r"$\alpha L=2(1+\beta)$")
ax.scatter([1.3, 3.6], [0.4, 0.4], c=[GREEN, PINK], s=52, zorder=5)
ax.annotate("устойчиво", (1.3, 0.4), xytext=(6, 7), textcoords="offset points")
ax.annotate("неустойчиво", (3.6, 0.4), xytext=(-70, 7), textcoords="offset points")
ax.set(xlim=(0, 4.15), ylim=(0, 1.02), xlabel=r"$\alpha L$",
       ylabel=r"$\beta$", title=r"Устойчивость для всех $\lambda\in(0,L]$")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.15)
save(fig, "stability_region")

r = 9 / 11
lambdas = np.linspace(1, 100, 300)
s = 1 + beta_hb - alpha_hb * lambdas
roots_plus = 0.5 * (s + np.lib.scimath.sqrt(s**2 - 4 * beta_hb))
roots_minus = 0.5 * (s - np.lib.scimath.sqrt(s**2 - 4 * beta_hb))
fig, ax = plt.subplots(figsize=(5.8, 3.5))
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(r * np.cos(t), r * np.sin(t), color=PALE, lw=2,
        label=r"$|z|=r$")
ax.scatter(roots_plus.real, roots_plus.imag, c=lambdas, cmap="viridis",
           s=8, label="корни")
ax.scatter(roots_minus.real, roots_minus.imag, c=lambdas, cmap="viridis", s=8)
ax.scatter([-r, r], [0, 0], c=PINK, marker="x", s=70, zorder=5)
ax.annotate(r"$\lambda=L$", (-r, 0), xytext=(-6, 9),
            textcoords="offset points", ha="right", fontsize=9)
ax.annotate(r"$\lambda=\mu$", (r, 0), xytext=(6, 9),
            textcoords="offset points", fontsize=9)
ax.axhline(0, color=GRAY, lw=0.6)
ax.axvline(0, color=GRAY, lw=0.6)
ax.set(xlim=(-1.05, 1.05), ylim=(-1.03, 1.03),
       xlabel=r"$\operatorname{Re} z$", ylabel=r"$\operatorname{Im} z$",
       title="При оптимальных параметрах все корни лежат на окружности")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="lower left", frameon=False)
save(fig, "root_circle")

kappa = np.logspace(0.03, 3, 250)
q_gd = (kappa - 1) / (kappa + 1)
q_hb = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.semilogx(kappa, q_gd, color=BLUE, lw=2.2, label="GD, лучший постоянный шаг")
ax.semilogx(kappa, q_hb, color=PINK, lw=2.2,
            label="тяжёлый шарик: асимптотический множитель")
ax.set(xlabel=r"число обусловленности $\kappa=L/\mu$",
       ylabel="множитель сокращения", ylim=(0, 1.02),
       title="Меньший множитель означает более быструю сходимость")
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.18)
save(fig, "rate_vs_condition")

gd_path = gd(A, x0, alpha_gd, 90)
hb_path = heavy_ball(A, x0, alpha_hb, beta_hb, 90)
norm_gd = np.linalg.norm(gd_path, axis=1)
norm_hb = np.linalg.norm(hb_path, axis=1)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.semilogy(norm_gd, color=BLUE, lw=2, label="GD, лучший постоянный шаг")
ax.semilogy(norm_hb, color=PINK, lw=2, label="тяжёлый шарик, параметры Поляка")
ax.set(xlabel="номер шага $k$", ylabel=r"расстояние $\|x_k-x^*\|_2$",
       title=r"Квадратичная функция: $\mu=1$, $L=100$")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.17, which="both")
save(fig, "error_comparison")

alpha, beta = 0.8, 0.5
path = heavy_ball(np.eye(1), [1], alpha, beta, 18)[:, 0]
vel = np.r_[0.0, np.diff(path)]
obj = 0.5 * path**2
energy = obj + beta / (2 * alpha) * vel**2
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.plot(obj, "-o", color=PINK, ms=3.5, label=r"функция $f(x_k)$")
ax.plot(energy, "-o", color=GREEN, ms=3.5, label=r"энергия $E_k$")
ax.set(xlabel="номер шага $k$", ylabel="значение",
       title="Энергия убывает, хотя значение функции колеблется")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.18)
save(fig, "objective_and_energy")

fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.35), sharex=True, sharey=True)
for ax, a, b, label in [
    (axes[0], 0.015, 0.1, "Слабая инерция"),
    (axes[1], alpha_hb, beta_hb, "Параметры Поляка"),
]:
    path = heavy_ball(A, x0, a, b, 5)
    quadratic_contours(ax, A)
    ax.plot(path[:, 0], path[:, 1], "-o", color=PINK, lw=1.7, ms=3.3)
    ax.scatter(path[0, 0], path[0, 1], marker="s", s=38, color=BLUE)
    ax.scatter(path[-1, 0], path[-1, 1], marker="o", s=48,
               facecolors="white", edgecolors=PINK, linewidths=1.6, zorder=7)
    ax.set_title(label)
save(fig, "momentum_parameter_paths")

print(f"Generated {len(list(OUT.glob('*.pdf')))} PDF figures in {OUT}")
