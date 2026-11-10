VENV ?= .venv
ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
VENV_PYTHON := $(VENV)/bin/python
endif
PYTHON ?= $(VENV_PYTHON)
BASE_PYTHON ?= python
export PYTHONPATH := src

.PHONY: setup test lint table sweep optional hf assets all clean-raw

setup:
	"$(BASE_PYTHON)" -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -e ".[dev]"

test:
	"$(PYTHON)" -m pytest -q

lint:
	"$(PYTHON)" -m ruff check .
	"$(PYTHON)" -m ruff format --check .

table:
	"$(PYTHON)" -m lora_lab.runner --config configs/exp01_param_table.yaml
	"$(PYTHON)" -m analysis.make_tables

sweep:
	"$(PYTHON)" -m lora_lab.runner --config configs/exp02_rank_sweep.yaml
	"$(PYTHON)" -m lora_lab.runner --config configs/exp03_fullrank_target.yaml
	"$(PYTHON)" -m lora_lab.runner --config configs/exp04_merge_check.yaml

optional:
	"$(PYTHON)" -m lora_lab.runner --config configs/exp05_alpha_scaling.yaml
	"$(PYTHON)" -m lora_lab.runner --config configs/exp06_qlora_sim.yaml

hf:
	"$(PYTHON)" -m lora_lab.runner --config configs/exp07_peft_distilbert.yaml

assets:
	"$(PYTHON)" -m analysis.make_tables
	"$(PYTHON)" -m analysis.make_figures
	"$(PYTHON)" -m analysis.make_summary
	"$(PYTHON)" report/worked_example.py

all: test lint table sweep assets

clean-raw:
	"$(PYTHON)" -c "import pathlib, shutil; p=pathlib.Path('results/raw'); shutil.rmtree(p) if p.exists() else None; p.mkdir(parents=True, exist_ok=True)"
