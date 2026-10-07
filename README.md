# Методы оптимизации в машинном обучении

Материалы курса **«Методы оптимизации в машинном обучении»** для совместной программы СМОЛ ГУ и ФКН НИУ ВШЭ, 2026/27 учебный год.

**Автор:** Никита Лукьяненко, старший преподаватель ФКН НИУ ВШЭ.

## Лекции

| № | Тема | PDF | Исходники |
|---|---|---|---|
| 1 | Оптимизация как язык машинного обучения | [PDF](pdf/lecture01_optimization_ml.pdf) | [LaTeX](lectures/lecture01/lecture01_optimization_ml.tex) |
| 2 | Геометрия оптимизации: производная, градиент и гессиан | [PDF](pdf/lecture02_geometry.pdf) | [LaTeX](lectures/lecture02/lecture02_geometry.tex) · [графики](lectures/lecture02/generate_figures.py) |
| 3 | Квадратичная оптимизация: геометрия, спектр и точное решение | [PDF](pdf/lecture03_quadratic.pdf) | [LaTeX](lectures/lecture03/lecture03_quadratic.tex) · [графики](lectures/lecture03/generate_figures.py) |
| 4 | Обусловленность и градиентный спуск: почему алгоритм сходится? | [PDF](pdf/lecture04_conditioning_gd.pdf) | [LaTeX](lectures/lecture04/lecture04_conditioning_gd.tex) · [графики](lectures/lecture04/generate_figures.py) |
| 5 | Выпуклость и условия оптимальности | [PDF](pdf/lecture05_convexity.pdf) | [LaTeX](lectures/lecture05/lecture05_convexity.tex) · [графики](lectures/lecture05/generate_figures.py) |
| 6 | Скорость сходимости градиентного спуска | [PDF](pdf/lecture06_convergence.pdf) | [LaTeX](lectures/lecture06/lecture06_convergence.tex) · [графики](lectures/lecture06/generate_figures.py) |

> PDF собираются автоматически GitHub Actions после изменений исходников. Если ссылки временно недоступны сразу после коммита, нужно дождаться завершения workflow **Build lecture PDFs**.

## Семинары

| № | Тема | Задачи |
|---|---|---|
| 2 | Геометрия оптимизации | [PDF](pdf/seminar02_theory.pdf) · [Colab](https://colab.research.google.com/github/Vault-Guy/optimization-methods-in-machine-learning/blob/main/seminars/seminar02/seminar02_computational_tasks.ipynb) |
| 3 | Квадратичная оптимизация, обусловленность и градиентный спуск | [PDF](pdf/seminar03_theory.pdf) · [Colab](https://colab.research.google.com/github/Vault-Guy/optimization-methods-in-machine-learning/blob/main/seminars/seminar03/seminar03_computational_tasks.ipynb) |
| 4 | Гладкость, выпуклость и условия оптимальности | [PDF](pdf/seminar04_theory.pdf) · [Colab](https://colab.research.google.com/github/Vault-Guy/optimization-methods-in-machine-learning/blob/main/seminars/seminar04/seminar04_computational_tasks.ipynb) |
## Домашние лабораторные

| № | Тема | Материалы |
|---|---|---|
| 1 | Геометрия задачи оптимизации и градиентные методы | [описание](labs/lab01/README.md) · [ноутбук](labs/lab01/lab01_geometry_gd.ipynb) · [открыть в Colab](https://colab.research.google.com/github/Vault-Guy/optimization-methods-in-machine-learning/blob/main/labs/lab01/lab01_geometry_gd.ipynb) |

Каждая лабораторная хранится в отдельном каталоге `labs/labXX/`; при необходимости внутри неё добавляются `data/` и `assets/`. Решения в публичный репозиторий не выкладываются.

## Структура репозитория

```text
lectures/
  lecture01/
  lecture02/
  lecture03/
  lecture04/
  lecture05/
  lecture06/
seminars/
  seminar02/
    seminar02_theory.tex
    seminar02_computational_tasks.ipynb
  seminar03/
    seminar03_theory.tex
    seminar03_computational_tasks.ipynb
  seminar04/
    README.md
    seminar04_theory.tex
    seminar04_computational_tasks.ipynb
labs/
  lab01/
    README.md
    lab01_geometry_gd.ipynb
pdf/                     # автоматически собранные лекции
.github/workflows/       # автоматическая сборка
```

## Сборка

Подробные инструкции находятся в [BUILD.md](BUILD.md).

Лекция 1 рассчитана на **XeLaTeX**, лекции 2–6 — на **pdfLaTeX**. Графики для лекций 2–6 генерируются соответствующими Python-скриптами. Автоматическая сборка сама получает HSE Beamer theme и сохраняет готовые PDF.

Python-зависимости перечислены в [requirements.txt](requirements.txt).

## Статус курса

Репозиторий обновляется в течение 2026/27 учебного года. Здесь хранятся актуальные версии уже подготовленных лекций и семинарских заданий; лабораторные и следующие темы будут добавляться по мере готовности.
