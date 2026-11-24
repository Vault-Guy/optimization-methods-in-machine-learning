# Сборка материалов

## Автоматическая сборка на GitHub

Workflow `.github/workflows/build-pdfs.yml` запускается автоматически при изменениях в:

- `lectures/**/*.tex`;
- `lectures/**/generate_figures.py`;
- `requirements.txt`;
- самом workflow.

Его также можно запустить вручную через **Actions → Build course PDFs → Run workflow**.

Workflow:

1. устанавливает Python-зависимости и TeX Live;
2. загружает HSE Beamer theme из зафиксированного коммита внешнего шаблона;
3. генерирует рисунки для лекций 2–11;
4. собирает лекцию 1 через XeLaTeX, лекции 2–11 через pdfLaTeX;
5. загружает одиннадцать PDF как GitHub Actions artifact;
6. при изменении PDF автоматически коммитит их в каталог `pdf/` ветки `main`.

Коммит, который обновляет только `pdf/*.pdf`, не запускает workflow повторно.

## Локальная сборка

Лекция 1 использует XeLaTeX. Лекции 2–11 используют pdfLaTeX.

Для лекций 2–11 сначала сгенерируйте рисунки соответствующими скриптами. Например, для лекции 11:

```bash
python lectures/lecture11/generate_figures.py
```

Python-зависимости:

```bash
pip install -r requirements.txt
```

Для локальной LaTeX-сборки необходим каталог `HSE-theme/` рядом с соответствующим `.tex`-файлом. Автоматическая сборка GitHub Actions получает тему самостоятельно и не требует хранить её копию в репозитории.

Примеры:

```bash
cd lectures/lecture01
latexmk -xelatex -interaction=nonstopmode -halt-on-error lecture01_optimization_ml.tex
```

```bash
cd lectures/lecture11
latexmk -pdf -interaction=nonstopmode -halt-on-error lecture11_heavy_ball.tex
```
