.PHONY: all sweep plot paper clean lint

# Fail loudly if required binaries are missing.
PYTHON   := python
JUPYTER  := jupyter
PDFLATEX := pdflatex
RUFF     := ruff
MYPY     := mypy

# Ordered list of notebooks to execute during sweep.
NOTEBOOKS := \
	notebooks/01_baseline.ipynb \
	notebooks/02_quantize_int8.ipynb \
	notebooks/03_quantize_int4.ipynb \
	notebooks/04_prune_structured.ipynb \
	notebooks/05_distill.ipynb \
	notebooks/06_combinations.ipynb

all: sweep plot paper

sweep: $(NOTEBOOKS)
	@echo "==> Running compression sweep (all six notebooks in order)"
	@command -v $(JUPYTER) >/dev/null 2>&1 || { echo "ERROR: jupyter not found. Install with: pip install jupyter"; exit 1; }
	@for nb in $(NOTEBOOKS); do \
		echo "  -> $$nb"; \
		$(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb" || { echo "ERROR: $$nb failed"; exit 1; }; \
	done
	@echo "==> Sweep complete. Results written to results/per_config/"

plot: results/compression_grid.csv src/plot_pareto.py
	@echo "==> Generating Pareto plot"
	@test -f results/compression_grid.csv || { echo "ERROR: results/compression_grid.csv not found. Run 'make sweep' first."; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: python not found."; exit 1; }
	$(PYTHON) src/plot_pareto.py
	@echo "==> Plot written to results/pareto_frontier.{png,pdf}"

paper: paper/note.tex
	@echo "==> Compiling paper/note.tex"
	@command -v $(PDFLATEX) >/dev/null 2>&1 || { echo "ERROR: pdflatex not found. Install texlive."; exit 1; }
	@test -f paper/note.tex || { echo "ERROR: paper/note.tex not found."; exit 1; }
	cd paper && $(PDFLATEX) -interaction=nonstopmode note.tex
	@echo "==> PDF written to paper/note.pdf"

clean:
	@echo "==> Cleaning build artefacts"
	rm -f paper/note.aux paper/note.log paper/note.out paper/note.pdf
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Clean complete (results/ and data/ are not removed; remove manually if needed)"

lint:
	@echo "==> Running ruff (linter)"
	@command -v $(RUFF) >/dev/null 2>&1 || { echo "ERROR: ruff not found. Install with: pip install ruff"; exit 1; }
	$(RUFF) check src/ scripts/*.py 2>/dev/null || $(RUFF) check src/
	@echo "==> Running mypy (strict type checking)"
	@command -v $(MYPY) >/dev/null 2>&1 || { echo "ERROR: mypy not found. Install with: pip install mypy"; exit 1; }
	$(MYPY) --strict src/
