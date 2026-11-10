# Required packages and engine notes

The fragment `results_discussion.tex` has no preamble and no `\documentclass`.
Add these packages to the main document preamble:

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
```

Notes:

- `booktabs` is required for `\toprule`, `\midrule`, and `\bottomrule`.
- `graphicx` is required for `\includegraphics` and for `\resizebox`, which is
  used by the two wide tables.
- `amsmath` and `amssymb` are required for `\mathbb{R}`, `\mathcal{N}`, and
  `\text`-free math notation used in the text.
- The prose is in English, so `pdflatex` is sufficient. No Vietnamese font or
  `T5` encoding is needed.
- No `hyperref` dependency is assumed; references use plain `\ref`.
- Figures are PNG copies of the generated `results/figures/*.png` files. The
  paths in the fragment are relative to the Overleaf project root:
  `figures/rank_sweep.png`, `figures/fullrank.png`, and
  `figures/alpha_scaling.png`.
- Tables are included with `\input{tables/params_memory}`,
  `\input{tables/rank_sweep}`, and `\input{tables/comparison}`. Each `.tex`
  file contains its own `table` environment, caption, and label.
- The bibliography keys used in the fragment are `hu2022lora` and
  `dettmers2023qlora`; merge `references_additions.bib` into the main `.bib`
  file.
