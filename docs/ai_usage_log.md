# AI usage log

Assignment: CS420 Seminar S08 - LoRA and QLoRA toy experiments
Date: 2026-11-08

## Tool and model

- Tool: opencode CLI (agentic coding assistant)
- Model: `opencode/deepseek-v4.1-flash-free`
- Environment: Windows 11, PowerShell 5.1, conda environment `deepfake`
  (Python 3.10.19, PyTorch 2.2.2+cu121, pytest 9.2.0, ruff 0.17.4)

## How AI was used

AI assistance was used for:

1. Scaffolding the repository structure, `pyproject.toml`, and `Makefile`.
2. Drafting and refactoring the registry, config loader, data generation,
   training loop, metrics, memory estimates, record schema, and runner.
3. Implementing the `full_ft` and `lora` methods, the `low_rank` and
   `decaying_spectrum` targets, and the identity and absmax-block quantizers.
4. Implementing experiments `exp01`-`exp07`, the analysis scripts, tests, and
   report generators.
5. Debugging test failures and analysis indexing issues.
6. Drafting the README, worked example, diagram, and this log.

The human author reviewed the code, chose the experiment scope and CPU budget,
approved the optional experiments, and verified the generated artifacts.

## Verification performed

- `pytest -q`: 50 tests passed.
- `ruff check .` and `ruff format --check .`: clean.
- Experiments `exp01`-`exp06` were executed with the recorded configs; every
  number in `results/` and `report/results_summary.md` comes from those CSV
  files.
- `analysis.make_tables`, `analysis.make_figures`, `analysis.make_summary`, and
  `report/worked_example.py` were rerun after the final experiment runs.
- The worked example asserts its hand-derived gradients against
  `torch.autograd` and asserts that the merged forward pass equals the
  low-rank forward pass.
- The extensibility claim was checked with a temporary one-file stub method
  plus one config line; the contract test passed and the stub was then removed.

`exp07` (PEFT DistilBERT/SST-2) was **not** run because it requires GPU/Colab
and optional Hugging Face dependencies. No DistilBERT/SST-2 metrics are claimed
anywhere in the report.

## Representative prompts

The human author used prompts equivalent to the following:

- "Implement the project described in `AGENT.md`, phase by phase, with tests
  before moving on. Keep every artifact in English."
- "Run the required experiments within a CPU budget and report the measured
  results, including results that contradict the expected ordering."
- "Generate the worked numerical example, diagram, README, and AI usage log
  from code or from the recorded results."
- "Do not fabricate metrics. Leave GPU-only experiments unrun and document
  that clearly."

## What AI did not do

- AI did not invent or edit experimental measurements by hand.
- AI did not run `exp07` or claim GPU results.
- AI did not replace the human review of the LoRA derivation, experiment
  design, or final report text.

## Suggested citation for the report

If the course requires citing AI assistance, the following wording can be
adapted:

> OpenAI-compatible opencode CLI with the `opencode/deepseek-v4.1-flash-free`
> model was used on 2026-11-08 to assist with code scaffolding, implementation,
> debugging, test authoring, and documentation. All experiments were executed
> locally, and all reported metrics were generated from the resulting CSV
> files. The authors reviewed and verified the code and results.

## Report section drafting (2026-11-08)

The AI assistant also drafted the LaTeX "Result and Discussion" fragment:

- Added `analysis/make_latex_tables.py`, which reads only
  `results/tables/*.csv` and `results/raw/*.csv` and generates the three LaTeX
  tables used in the section.
- Created `report/latex/` with the section fragment, generated tables, figure
  copies, bibliography additions, package notes, number-source mapping, and a
  short TODO list.
- Explicitly omitted exp07 (PEFT DistilBERT/SST-2) at the author's request and
  added a limitation sentence explaining that it requires a GPU and the
  optional Hugging Face dependencies.
- Reported the measured exp02 and exp03 orderings exactly as recorded, including
  the cases where LoRA was better than full fine-tuning, and flagged them as
  toy optimisation effects rather than general claims.
- Every prose number is mapped to its CSV row in
  `report/latex/number_sources.md`.
- Compile check: `pdflatex` + `bibtex` on a throwaway wrapper outside the
  deliverable folder. The wrapper produced 6 pages including the title page and
  references, with no LaTeX warnings or errors in the final pass. The section
  itself is about four pages.

The human author remains responsible for independently verifying the claims,
the citations, and the final page budget in the main Overleaf document.
