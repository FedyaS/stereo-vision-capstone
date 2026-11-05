# Documentation

| | |
|---|---|
| [architecture.md](architecture.md) | **What runs where.** The four processes and their ports, the module map, and how one benchmark cell flows from dataset loader to a row of `results.csv`. With diagrams. Start here to understand the code. |
| [repository.md](repository.md) | **What is committed.** The full directory tree with per-directory git status, every ignore rule and the reason for it, and how to commit a result that matters without `git add -f`. Start here to understand the layout. |
| [figures/](figures/) | Scratch space for figures kept for write-ups. Benchmark runs write to `results/<timestamp>/figures/` instead. |
| [presentation/](presentation/) | The LaTeX beamer midterm presentation, `main.tex` and its PDF. |

Elsewhere in the repository:

| | |
|---|---|
| [../README.md](../README.md) | What the project is, the quickstart, the algorithms and datasets, how the numbers are produced, the tuning protocol, the results, and the caveats. |
| [../app/README.md](../app/README.md) | The explorer's four tabs and what to try in each. |
| [../data/README.md](../data/README.md) | The Middlebury dataset: what the download contains, which scenes are deliberately awkward. |
| [../models/README.md](../models/README.md) | The ONNX weights and where they come from. |
| [../results/keep/README.md](../results/keep/README.md) | The committed runs, and why each is kept. |
| [../notebooks/](../notebooks/) | Using `bench` from a notebook, and the original hand-written matcher the harness grew out of. |

## Building the presentation

```bash
cd docs/presentation && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`main.pdf` is committed; the `.aux`, `.bbl`, `.nav` and friends are not, so a
build leaves the working tree clean apart from the PDF.
