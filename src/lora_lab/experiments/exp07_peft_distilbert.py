"""exp07 - optional PEFT LoRA on DistilBERT/SST-2.

Hypothesis: PEFT LoRA adapts a pretrained transformer with far fewer trainable
parameters than full fine-tuning while retaining most SST-2 accuracy. Setup:
``peft`` LoRA ranks 4 and 16 on ``distilbert-base-uncased`` with the GLUE SST-2
split. Expected outcome: accuracy within a small margin of full fine-tuning at
a small fraction of trainable parameters.

This experiment requires the optional ``hf`` dependencies and a GPU/Colab. It
is intentionally not run by ``make all`` and was left unrun in this environment.
"""

from __future__ import annotations

import time
from typing import Any

from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register


@register(EXPERIMENTS, "exp07_peft_distilbert")
class PeftDistilBertExperiment(Experiment):
    """PEFT LoRA fine-tuning on DistilBERT/SST-2 (optional, GPU recommended)."""

    name = "exp07_peft_distilbert"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        try:
            import datasets
            import numpy as np
            import torch
            import transformers
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "exp07 requires the optional hf extras: pip install -e .[hf]"
            ) from exc

        if not torch.cuda.is_available() and not bool(cfg.get_extra("allow_cpu", False)):
            raise RuntimeError(
                "exp07 requires a GPU/Colab by default; set extra.allow_cpu: true to force CPU"
            )

        model_name = str(cfg.require_extra("model_name", str))
        dataset_name = str(cfg.require_extra("dataset_name", str))
        max_length = int(cfg.require_extra("max_length", int))
        data = datasets.load_dataset("glue", dataset_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = data.map(tokenize, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        records: list[RunRecord] = []
        for method_spec in cfg.methods:
            for hparams in method_spec.expand():
                for seed in cfg.seeds:
                    records.append(
                        self._run_one(
                            cfg=cfg,
                            hparams=hparams,
                            seed=seed,
                            model_name=model_name,
                            tokenized=tokenized,
                            transformers=transformers,
                            torch=torch,
                            numpy=np,
                            LoraConfig=LoraConfig,
                            TaskType=TaskType,
                            get_peft_model=get_peft_model,
                        )
                    )
        return records

    @staticmethod
    def _run_one(
        *,
        cfg: ExperimentConfig,
        hparams: dict[str, Any],
        seed: int,
        model_name: str,
        tokenized: Any,
        transformers: Any,
        torch: Any,
        numpy: Any,
        LoraConfig: Any,
        TaskType: Any,
        get_peft_model: Any,
    ) -> RunRecord:
        transformers.set_seed(seed)
        base = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(hparams["r"]),
            lora_alpha=float(hparams["alpha"]),
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none",
        )
        model = get_peft_model(base, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        lr = float(cfg.train.lr_grid[0])

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = numpy.argmax(logits, axis=-1)
            return {"accuracy": float((predictions == labels).mean())}

        arguments = transformers.TrainingArguments(
            output_dir=f"results/raw/exp07_tmp_seed{seed}",
            num_train_epochs=cfg.train.epochs,
            per_device_train_batch_size=cfg.train.batch_size,
            per_device_eval_batch_size=cfg.train.batch_size,
            learning_rate=lr,
            seed=seed,
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = transformers.Trainer(
            model=model,
            args=arguments,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )
        started = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - started
        val_metrics = trainer.evaluate(tokenized["validation"])
        test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

        return RunRecord(
            experiment="exp07_peft_distilbert",
            method="lora_peft",
            method_hparams=dict(hparams),
            target="sst2",
            target_cfg={"model_name": model_name},
            quantizer="none",
            d=cfg.data.d,
            k=cfg.data.k,
            seed=seed,
            lr=lr,
            epochs=cfg.train.epochs,
            trainable_params=trainable,
            total_params=total,
            trainable_ratio=trainable / total,
            train_mse=None,
            val_mse=None,
            test_mse=None,
            rel_delta_err=None,
            train_time_s=elapsed,
            extra={
                "val_accuracy": float(val_metrics.get("eval_accuracy", float("nan"))),
                "test_accuracy": float(test_metrics.get("test_accuracy", float("nan"))),
                "requires_gpu": True,
                "optional_hf_extra": True,
            },
        )
