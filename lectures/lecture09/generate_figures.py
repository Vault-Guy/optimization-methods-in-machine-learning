"""Generate all lecture 9 illustrations with Python/matplotlib.

Run from any directory: python generate_figures.py
The PDF figures are included by the Beamer source; PNG copies aid review.
"""
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def clean(ax):
    ax.grid(alpha=0.18)
    ax.axhline(0, color=GRAY, lw=0.7, alpha=0.6)


# 1. Local model: generous labels and a legend above the plotting area.
x = np.linspace(-0.75, 1.75, 600)
x0 = 0.7
f = lambda z: 0.18 * z**4 + 0.32 * z**2 + 0.10 * z
g0 = 0.72 * x0**3 + 0.64 * x0 + 0.10
h0 = 2.16 * x0**2 + 0.64
linear = f(x0) + g0 * (x - x0)
quad = linear + 0.5 * h0 * (x - x0)**2
fig, ax = plt.subplots(figsize=(9.0, 3.35))
ax.axvspan(x0 - 0.42, x0 + 0.42, color="#E8F0F7", zorder=0)
ax.plot(x, f(x), color=BLUE, lw=2.8, label="исходная функция")
ax.plot(x, linear, color=GRAY, lw=2.2, ls="--", label="линейная модель")
ax.plot(x, quad, color=RED, lw=2.5, label="квадратичная модель")
ax.scatter([x0], [f(x0)], color=BLUE, edgecolor="white", linewidth=1.1, s=90, zorder=5)
ax.axvline(x0, color=PALE, lw=1.1, ls=":")
ax.annotate(r"точка $x$", (x0, f(x0)), xytext=(12, -25), textcoords="offset points", fontsize=11, color=BLUE)
ax.set(xlim=(-0.7, 1.7), ylim=(-1.0, 2.7), xlabel="параметр", ylabel="значение")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=11)
clean(ax)
save(fig, "local_models")


# 2. Ellipses and different directions at the same point.
A = np.array([[6.0, 1.5], [1.5, 1.1]])
xx, yy = np.meshgrid(np.linspace(-2.3, 2.3, 280), np.linspace(-2.1, 2.1, 280))
zz = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2)
p = np.array([1.65, 0.55])
g = A @ p
dgrad = -g / np.linalg.norm(g) * 0.9
dnewton = -p
fig, ax = plt.subplots(figsize=(6.7, 3.8))
ax.contour(xx, yy, zz, levels=[0.4, 1, 2, 4, 7, 11], colors=PALE, linewidths=1.2)
ax.scatter(*p, color=BLUE, s=50, zorder=6)
ax.scatter([0], [0], color=GREEN, s=55, zorder=6)
ax.arrow(*p, *dgrad, color=RED, width=0.018, head_width=0.13, length_includes_head=True)
ax.arrow(*p, *dnewton, color=GREEN, width=0.018, head_width=0.13, length_includes_head=True)
ax.text(0.95, 0.13, "антиградиент", color=RED)
ax.text(0.30, 0.74, "шаг Ньютона", color=GREEN)
ax.set(xlim=(-2, 2.2), ylim=(-1.5, 1.65), aspect="equal", xlabel=r"$x_1$", ylabel=r"$x_2$")
save(fig, "directions_contours")


# 3. Newton reaches the minimum of a positive-definite quadratic in one step.
A = np.diag([8.0, 1.0])
p = np.array([1.65, 1.35])
g = A @ p
alpha = (g @ g) / (g @ A @ g)
gd1 = p - alpha * g
gd2g = A @ gd1
gd2 = gd1 - (gd2g @ gd2g) / (gd2g @ A @ gd2g) * gd2g
xx, yy = np.meshgrid(np.linspace(-2.1, 2.1, 280), np.linspace(-1.9, 1.9, 280))
zz = 0.5 * (8 * xx**2 + yy**2)
fig, ax = plt.subplots(figsize=(7.0, 3.7))
ax.contour(xx, yy, zz, levels=[0.35, 1, 2, 4, 7, 11], colors=PALE)
ax.plot([p[0], gd1[0], gd2[0]], [p[1], gd1[1], gd2[1]], "o-", color=RED, lw=2, label="градиентный спуск")
ax.plot([p[0], 0], [p[1], 0], "o-", color=GREEN, lw=2.3, label="метод Ньютона")
ax.scatter([0], [0], color=BLUE, s=50, zorder=6)
ax.legend(frameon=False, loc="lower left")
ax.set(xlim=(-1.7, 2), ylim=(-1.65, 1.7), aspect="equal", xlabel=r"$x_1$", ylabel=r"$x_2$")
save(fig, "quadratic_one_step")


# 4. Full Newton step can fail although f is strictly convex everywhere.
f = lambda z: np.sqrt(1 + z**2)
x = np.linspace(-9, 3, 600)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.45), gridspec_kw={"width_ratios": [1.55, 0.8]})
ax = axs[0]
ax.plot(x, f(x), color=BLUE, lw=2.8)
ax.scatter([2, -8], [f(2), f(-8)], color=[GREEN, RED], edgecolor="white", linewidth=1.2, s=105, zorder=5)
ax.annotate("", xy=(-8, 8.8), xytext=(2, 8.8), arrowprops=dict(arrowstyle="->", lw=1.8, color=RED))
ax.text(-5.7, 9.04, r"полный шаг: $2\longrightarrow-8$", color=RED, fontsize=11)
ax.annotate(r"$x_0=2$", (2, f(2)), xytext=(-58, 17), textcoords="offset points", color=GREEN, fontsize=11)
ax.annotate(r"$x_1=-8$", (-8, f(-8)), xytext=(10, -22), textcoords="offset points", color=RED, fontsize=11)
ax.set(xlim=(-9, 3), ylim=(0.75, 9.5), xlabel=r"$x$", ylabel=r"$f(x)=\sqrt{1+x^2}$")
ax.grid(alpha=0.14)
ax = axs[1]
bars = ax.bar([0, 1], [f(2), f(-8)], color=[GREEN, RED], width=0.52, alpha=0.9)
ax.set_xticks([0, 1], [r"$f(x_0)$", r"$f(x_1)$"])
ax.set_ylim(0, 9.5)
ax.set_ylabel("значение функции")
ax.set_title("Функция выросла", color=RED, fontsize=12)
for bar, value in zip(bars, [f(2), f(-8)]):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.25, f"{value:.2f}", ha="center", fontsize=11)
ax.grid(axis="y", alpha=0.14)
ax.set_axisbelow(True)
save(fig, "full_step_failure")


# 5. Armijo test for the same example; direct labels and accepted region.
x0 = 2.0
p = -x0 * (1 + x0**2)
g = x0 / np.sqrt(1 + x0**2)
alpha = np.linspace(0, 1.05, 500)
phi = f(x0 + alpha * p)
c = 0.1
armijo = f(x0) + c * alpha * g * p
fig, ax = plt.subplots(figsize=(9.0, 3.55))
ax.fill_between(alpha, phi, armijo, where=phi <= armijo, interpolate=True,
                color=GREEN, alpha=0.15, label="шаг удовлетворяет Армихо")
ax.plot(alpha, phi, color=BLUE, lw=2.7)
ax.plot(alpha, armijo, color=RED, ls="--", lw=2.0)
for a, accepted in [(1.0, False), (0.5, False), (0.25, True)]:
    yy = f(x0 + a * p)
    ax.scatter([a], [yy], color=GREEN if accepted else RED,
               edgecolor="white", linewidth=1.1, s=95, zorder=5)
ax.text(0.27, 1.43, r"$\alpha=0.25$: принят", color=GREEN, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
ax.text(0.46, 3.78, r"$\alpha=0.5$: отказ", color=RED, fontsize=10)
ax.text(0.77, 8.28, r"$\alpha=1$: отказ", color=RED, fontsize=10)
ax.text(0.64, 5.75, r"$f(x+\alpha p)$", color=BLUE, fontsize=12)
ax.text(0.71, 1.12, "порог Армихо", color=RED, fontsize=11)
ax.set(xlim=(0, 1.05), ylim=(0.55, 9.15), xlabel=r"доля полного шага $\alpha$", ylabel="значение")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(alpha=0.14)
save(fig, "armijo_example")


# 6. Damped Newton iterates and objective values on the globalized example.
def damped_step(z, beta=0.5, c=0.1):
    p = -z * (1 + z**2)
    gp = z / np.sqrt(1 + z**2) * p
    a = 1.0
    while f(z + a * p) > f(z) + c * a * gp:
        a *= beta
    return z + a * p, a


xs = [2.0]
alphas = []
for _ in range(5):
    z, a = damped_step(xs[-1])
    xs.append(z)
    alphas.append(a)
fig, axs = plt.subplots(1, 2, figsize=(9.3, 3.15))
k = np.arange(len(xs))
axs[0].plot(k, xs, "o-", color=GREEN, lw=2)
axs[0].axhline(0, color=GRAY, lw=0.8)
axs[0].set(xlabel="итерация", ylabel=r"$x_k$", xticks=k)
axs[1].plot(k, f(np.array(xs)) - 1, "o-", color=BLUE, lw=2)
axs[1].set(xlabel="итерация", ylabel=r"$f(x_k)-f(0)$", xticks=k)
for ax in axs:
    ax.grid(alpha=0.18)
save(fig, "damped_progress")


# 7. Singular Hessian: Newton is only linearly convergent on x^4.
k = np.arange(9)
fig, ax = plt.subplots(figsize=(6.9, 3.3))
ax.semilogy(k, (2 / 3) ** k, "o-", color=RED, lw=2)
ax.set(xlabel="итерация", ylabel=r"$|x_k|$ при $x_0=1$", xticks=k)
ax.grid(alpha=0.2)
save(fig, "singular_linear_rate")

print(f"Generated {len(list(OUT.glob('*.pdf')))} PDF figures in {OUT}")
print("Damped example x_k:", [round(z, 8) for z in xs])
print("Accepted alpha_k:", alphas)
