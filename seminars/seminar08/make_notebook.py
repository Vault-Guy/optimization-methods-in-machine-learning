"""Build the student Colab notebook for seminar 8 from readable cell sources."""
from pathlib import Path
import json

cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().splitlines(keepends=True)})


markdown(r"""
# Семинар 8. Предобусловливание: вычислительные задачи

Материал лекции 10. Каждая программная задача соответствует задаче с тем же номером в бумажном листке. Задачи **1–5** — основные, **6–10** — дополнительные. Данные создаются внутри ноутбука, скачивание не требуется.

**Работа:** заполните `TODO`, выполните расчёты и после каждого эксперимента запишите короткий вывод. Для шага $Pp=-g$ применяйте `np.linalg.solve(P, -g)`, а не формируйте $P^{-1}$. Сравнивайте методы при одинаковой начальной точке и явно подписывайте норму ошибки на графике.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(8)
plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True, "grid.alpha": 0.2})
print("NumPy:", np.__version__)
""")

markdown(r"""
## Основные задачи

### Задача 1. Замена координат

Аналог задачи 1 листка. Реализуйте $q(x)$, $\nabla q(x)$ и градиент $\widetilde q(y)=q(Sy)$ по правилу цепочки. Выполните один шаг в $y$ и соответствующий предобусловленный шаг в $x$. Проверьте совпадение результатов численно. Постройте линии уровня исходной функции с отмеченными точками $x_0$, $x_1$, $x^*$; поясните, почему в новых координатах линии уровня имеют другую форму.
""")

code(r"""
A1 = np.diag([1.0, 16.0])
b1 = np.array([1.0, 4.0])
S1 = np.diag([1.0, 0.25])
x01 = np.zeros(2)
y01 = np.zeros(2)

def q1(x):
    return 0.5 * x @ A1 @ x - b1 @ x

def grad_x1(x):
    # TODO: аналитический градиент q1
    return None

def grad_y1(y):
    # TODO: применить правило цепочки к q1(S1 @ y)
    return None

if grad_x1(x01) is not None and grad_y1(y01) is not None:
    # TODO: вычислить P1, один шаг в y, соответствующий x1 и шаг P-GD в x
    P1, y11, x11, x11_pgd = None, None, None, None
    if x11 is not None and x11_pgd is not None:
        xstar1 = np.linalg.solve(A1, b1)
        print("ошибка совпадения шагов:", np.linalg.norm(x11 - x11_pgd))
        assert np.allclose(x11, x11_pgd)
        grid = np.linspace(-0.5, 1.5, 120)
        Xg, Yg = np.meshgrid(grid, grid)
        Z = np.array([q1(np.array([u, v])) for u, v in zip(Xg.flat, Yg.flat)]).reshape(Xg.shape)
        plt.contour(Xg, Yg, Z, levels=16)
        for point, label in [(x01, "start"), (x11, "step"), (xstar1, "minimum")]:
            plt.scatter(*point, label=label)
        plt.axis("equal"); plt.legend(); plt.show()
""")

markdown(r"""
### Задача 2. Направление спуска

Аналог задачи 2 листка. Решите $Pp=-g$, вычислите $g^\top p$ и $p^\top Pp$. Затем повторите опыт для 100 случайных положительно определённых матриц $P=C^\top C+0.1I$ и ненулевых $g$. Постройте гистограмму $g^\top p$ и объясните, почему все значения имеют один знак. Для численной проверки используйте небольшой допуск.
""")

code(r"""
P2 = np.array([[2.0, 1.0], [1.0, 2.0]])
g2 = np.array([2.0, -1.0])

def direction2(P, g):
    # TODO: решить P @ p = -g
    return None

if direction2(P2, g2) is not None:
    p2 = direction2(P2, g2)
    print("p:", p2, "gTp:", g2 @ p2, "pTPp:", p2 @ P2 @ p2)
    slopes2 = []
    for _ in range(100):
        C = rng.normal(size=(3, 3))
        P = C.T @ C + 0.1*np.eye(3)
        g = rng.normal(size=3)
        slopes2.append(g @ direction2(P, g))
    slopes2 = np.asarray(slopes2)
    assert np.max(slopes2) < 1e-12
    plt.hist(slopes2, bins=20); plt.xlabel(r"$g^\top p$"); plt.show()
""")

markdown(r"""
### Задача 3. Спектр и скорость

Аналог задачи 3 листка. Найдите спектры $A$ и $B=P^{-1/2}AP^{-1/2}$. Вычислите оптимальные постоянные шаги для GD и P-GD. Из $x_0=(1,1)$ сделайте 25 шагов обоими методами и постройте $\|x_k\|_P$ в логарифмическом масштабе. Сравните экспериментальный множитель с теоретическим; объясните, почему для P-GD здесь возможны смены знаков координат.
""")

code(r"""
A3 = np.diag([1.0, 9.0])
P3 = np.diag([1.0, 3.0])
x03 = np.ones(2)

def run3(A, P, x0, alpha, steps):
    # TODO: вернуть массив x_0,...,x_steps для x_next=x-alpha*solve(P,A@x)
    return None

# TODO: найти спектры A3 и B3, шаги alpha_gd и alpha_pgd
eigsA3, eigsB3, alpha_gd3, alpha_pgd3 = None, None, None, None
if alpha_gd3 is not None and alpha_pgd3 is not None:
    gd3 = run3(A3, np.eye(2), x03, alpha_gd3, 25)
    pgd3 = run3(A3, P3, x03, alpha_pgd3, 25)
    if gd3 is not None and pgd3 is not None:
        pnorm = lambda xs: np.sqrt(np.einsum("ni,ij,nj->n", xs, P3, xs))
        plt.semilogy(pnorm(gd3), label="GD")
        plt.semilogy(pnorm(pgd3), label="P-GD")
        plt.xlabel("итерация"); plt.ylabel(r"$\|x_k\|_P$"); plt.legend(); plt.show()
        print("спектры:", eigsA3, eigsB3)
""")

markdown(r"""
### Задача 4. Когда Якоби помогает

Аналог задачи 4 листка. Для обеих матриц вычислите $P_J=\operatorname{diag}(A)$ и спектр $B=P_J^{-1/2}AP_J^{-1/2}$. Сравните числа обусловленности до и после. Постройте рядом изображения матриц $A$ и $B$ (`imshow`): какая связь между координатами не исчезает после диагонального масштабирования? Дополнительно проверьте, что при $P=A$ матрица $B$ единичная.
""")

code(r"""
A4_list = [np.diag([1.0, 9.0]), np.array([[5.0, 4.0], [4.0, 5.0]])]

def jacobi_spectrum4(A):
    # TODO: вернуть P_J, B и собственные значения B
    return None

for number, A in enumerate(A4_list, start=1):
    result = jacobi_spectrum4(A)
    if result is not None:
        P, B, eigB = result
        print(number, "cond(A)=", np.linalg.cond(A), "cond(B)=", np.max(eigB)/np.min(eigB))
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.7))
        axes[0].imshow(A, cmap="Blues"); axes[0].set_title("A")
        axes[1].imshow(B, cmap="Blues"); axes[1].set_title("B")
        plt.show()
""")

markdown(r"""
### Задача 5. Ridge и предобусловленный шаг

Аналог задачи 5 листка. Для заданных $X,y$ реализуйте значение $F$, градиент и два алгоритма: GD с шагом $1/L$ и P-GD с $P=\operatorname{diag}(A)$, $\alpha=1$. Стартуйте из нуля и постройте разность $F(w_k)-F(w^*)$ за 12 шагов. Проверьте значение первого шага, затем замените диагональную $X$ на слегка коррелированную и исследуйте, что изменилось.
""")

code(r"""
X5 = np.diag([1.0, 4.0])
y5 = np.ones(2)
w05 = np.zeros(2)

def ridge5(w, X=X5, y=y5):
    residual = X @ w - y
    return 0.5 * residual @ residual + 0.5 * w @ w

def ridge_grad5(w, X=X5, y=y5):
    # TODO: градиент Ridge при коэффициенте регуляризации 1
    return None

def run5(X, y, P, alpha, steps=12):
    # TODO: выполнить steps шагов из w=0 и вернуть массив траектории
    return None

if ridge_grad5(w05) is not None:
    A5 = X5.T @ X5 + np.eye(2)
    b5 = X5.T @ y5
    wstar5 = np.linalg.solve(A5, b5)
    P5 = np.diag(np.diag(A5))
    gd5 = run5(X5, y5, np.eye(2), 1/np.linalg.eigvalsh(A5)[-1])
    pgd5 = run5(X5, y5, P5, 1.0)
    if gd5 is not None and pgd5 is not None:
        plt.semilogy([max(ridge5(w)-ridge5(wstar5), 1e-16) for w in gd5], label="GD")
        plt.semilogy([max(ridge5(w)-ridge5(wstar5), 1e-16) for w in pgd5], label="P-GD")
        plt.xlabel("итерация"); plt.ylabel(r"$F(w_k)-F(w^*)$"); plt.legend(); plt.show()
    # TODO: повторить для X с ненулевыми внедиагональными элементами
""")

markdown(r"""
## Дополнительные задачи

### Задача 6. Неопределённый и вырожденный предобусловливатель

Аналог задачи 6 листка. Посчитайте направление и изменение $f(x+\alpha p)$ при нескольких положительных $\alpha$ для $P=\operatorname{diag}(1,-1)$. Постройте зависимость от $\alpha$. Проверьте `np.linalg.solve` для вырожденного $P=\operatorname{diag}(1,0)$ и объясните исключение без отключения ячеек ноутбука.
""")

code(r"""
x6 = np.array([0.0, 1.0])
P6_bad = np.diag([1.0, -1.0])
P6_singular = np.diag([1.0, 0.0])
f6 = lambda x: 0.5 * x @ x
g6 = x6.copy()

# TODO: решить систему для P6_bad и найти p6
p6 = None
if p6 is not None:
    aa = np.linspace(0, 2, 100)
    plt.plot(aa, [f6(x6+a*p6) for a in aa], label=r"$f(x+\alpha p)$")
    plt.axhline(f6(x6), color="gray", linestyle="--", label=r"$f(x)$")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.show()
    print("производная по направлению:", g6 @ p6)

try:
    # TODO: попробовать решить P6_singular @ p = -g6
    pass
except np.linalg.LinAlgError as exc:
    print("У вырожденной системы нет единственного решения:", exc)
""")

markdown(r"""
### Задача 7. Спектральная эквивалентность

Аналог задачи 7 листка. Для диагонального примера проверьте $aP\preceq A\preceq bP$ по собственным значениям разностей. Затем сгенерируйте 20 случайных трёхмерных матриц $B$ с собственными значениями в $[a,b]$, положите $A=P^{1/2}BP^{1/2}$ и убедитесь, что оценка $\kappa(B)\le b/a$ выполняется. Сравните теоретический и фактический худший множитель при $\alpha=2/(a+b)$.
""")

code(r"""
P7 = np.diag([2.0, 8.0])
A7 = np.diag([1.0, 16.0])
a7, b7 = 0.5, 2.0
P7_random = np.diag([1.0, 2.0, 5.0])

def spectrum7(A, P):
    # TODO: построить B=P^{-1/2} A P^{-1/2} для диагонального P и вернуть eigvalsh(B)
    return None

if spectrum7(A7, P7) is not None:
    print("спектр B:", spectrum7(A7, P7))
    print("нижняя PSD граница:", np.linalg.eigvalsh(A7-a7*P7))
    print("верхняя PSD граница:", np.linalg.eigvalsh(b7*P7-A7))
    # TODO: сгенерировать случайные ортогональные Q размерности 3 и B=Q diag(lambda) Q.T,
    #      затем A=P7_random^{1/2} B P7_random^{1/2} и проверить спектральные границы
""")

markdown(r"""
### Задача 8. Плохой выбор $P$

Аналог задачи 8 листка. Сравните исходный спектр и спектр после $P=\operatorname{diag}(1,64)$. Запустите оба метода с оптимальными постоянными шагами из $x_0=(1,1)$ на 30 итераций. На одном графике покажите $\|x_k\|_2$, на другом — отношение соседних ошибок. Сопоставьте наблюдения с худшими теоретическими множителями.
""")

code(r"""
A8 = np.diag([1.0, 4.0])
P8 = np.diag([1.0, 64.0])
x08 = np.ones(2)

def run8(A, P, x0, alpha, steps=30):
    # TODO: вернуть массив x_0,...,x_steps
    return None

# TODO: вычислить B8, оптимальные alpha для GD и P-GD
B8, alpha_gd8, alpha_pgd8 = None, None, None
if alpha_gd8 is not None and alpha_pgd8 is not None:
    gd8 = run8(A8, np.eye(2), x08, alpha_gd8)
    pgd8 = run8(A8, P8, x08, alpha_pgd8)
    if gd8 is not None and pgd8 is not None:
        for trajectory, label in [(gd8, "GD"), (pgd8, "P-GD")]:
            errors = np.linalg.norm(trajectory, axis=1)
            plt.semilogy(errors, label=label)
        plt.xlabel("итерация"); plt.ylabel(r"$\|x_k\|_2$"); plt.legend(); plt.show()
        # TODO: построить отношение ||x_{k+1}||/||x_k||, пропуская нулевые знаменатели
""")

markdown(r"""
### Задача 9. Масштабирование признаков и штраф

Аналог задачи 9 листка. Реализуйте исходную функцию $F(w)$ и две функции в координатах $z$: правильную $F(Sz)$ и ошибочную, где штраф заменён на $\|z\|^2/2$. Найдите минимизаторы, переведите их обратно в координаты $w$ и сравните значения исходной функции. Постройте сечения двух функций по $z_2$ при фиксированном $z_1=1/2$.
""")

code(r"""
X9 = np.diag([1.0, 2.0])
y9 = np.ones(2)
S9 = np.diag([1.0, 0.5])

def original9(w):
    residual = X9 @ w - y9
    return 0.5*residual @ residual + 0.5*w @ w

def transformed9(z):
    # TODO: вернуть original9(S9 @ z)
    return None

def wrong9(z):
    # TODO: заменить штраф 0.5*||S9 @ z||^2 на 0.5*||z||^2
    return None

if transformed9(np.zeros(2)) is not None and wrong9(np.zeros(2)) is not None:
    # TODO: решить нормальные уравнения для правильной и ошибочной функций
    zstar9, zwrong9 = None, None
    if zstar9 is not None and zwrong9 is not None:
        print("w из верной задачи:", S9 @ zstar9)
        print("w из ошибочной задачи:", S9 @ zwrong9)
        print("исходные потери:", original9(S9 @ zstar9), original9(S9 @ zwrong9))
        zz = np.linspace(-0.5, 1.5, 180)
        plt.plot(zz, [transformed9(np.array([0.5, t])) for t in zz], label="верный штраф")
        plt.plot(zz, [wrong9(np.array([0.5, t])) for t in zz], label="изменённый штраф")
        plt.xlabel(r"$z_2$"); plt.legend(); plt.show()
""")

markdown(r"""
### Задача 10. Логистическая регрессия

Аналог задачи 10 листка. Вычислите градиент и гессиан логистической потери с $\ell_2$-регуляризацией. В нуле найдите диагональный шаг $Pp=-g$, $P=\operatorname{diag}(H)$, и точное ньютоновское направление $Hp_N=-g$. Сравните производные по направлениям и значения $F(\alpha p)$ для нескольких небольших $\alpha$. Объясните, какая информация о взаимосвязи признаков теряется при диагональном $P$.
""")

code(r"""
X10 = np.array([[1.0, 0.0], [1.0, 1.0]])
y10 = np.array([1.0, -1.0])
w010 = np.zeros(2)

def logistic10(w):
    margins = y10 * (X10 @ w)
    return np.logaddexp(0.0, -margins).sum() + 0.5*w @ w

def grad10(w):
    # TODO: аналитический градиент
    return None

def hess10(w):
    # TODO: аналитический гессиан
    return None

if grad10(w010) is not None and hess10(w010) is not None:
    g10, H10 = grad10(w010), hess10(w010)
    P10 = np.diag(np.diag(H10))
    p_diag10 = np.linalg.solve(P10, -g10)
    p_newton10 = np.linalg.solve(H10, -g10)
    print("градиент:", g10, "гессиан:\n", H10)
    print("диагональное направление:", p_diag10, "точное ньютоновское:", p_newton10)
    aa = np.linspace(0, 1, 150)
    plt.plot(aa, [logistic10(a*p_diag10) for a in aa], label="диагональное P")
    plt.plot(aa, [logistic10(a*p_newton10) for a in aa], label="полный H")
    plt.xlabel(r"$\alpha$"); plt.ylabel(r"$F(\alpha p)$"); plt.legend(); plt.show()
""")

markdown(r"""
---

## Что сдавать

Для каждой выполненной задачи сохраните код, численные результаты или графики и краткий вывод: какой результат лекции используется, какие его предпосылки выполнены и что можно заключить из опыта. Дополнительные задачи можно выполнить после основных.
""")

for index, cell in enumerate(cells, 1):
    cell["id"] = f"s08-{index:03d}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
output = Path(__file__).resolve().parent / "seminar08_computational_tasks.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {output} with {len(cells)} cells")
