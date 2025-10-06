from __future__ import annotations

import argparse
import os
import random

import numpy as np
from accelerate import Accelerator
from datasets import DatasetDict
from evaluate import load as load_metric
from transformers import (
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Trainer,
	TrainingArguments,
	set_seed,
)

from ets_translator.utils.config import TranslationConfig
from ets_translator.data.datasets import load_translation_dataset, build_preprocess_functions


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train translation model")
	parser.add_argument("--config", type=str, required=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = TranslationConfig.load(args.config)

	set_seed(cfg.seed)

	tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

	ds = load_translation_dataset(cfg.hf_dataset.model_dump())
	# Expect splits available: train/validation/test depending on dataset
	train_split = cfg.hf_dataset.split_train
	eval_split = cfg.hf_dataset.split_eval

	preprocess = build_preprocess_functions(
		tokenizer,
		source_lang=cfg.source_lang,
		target_lang=cfg.target_lang,
		max_source_length=cfg.max_source_length,
		max_target_length=cfg.max_target_length,
	)

	processed = {}
	for split in [train_split, eval_split]:
		if split in ds:
			processed[split] = ds[split].map(preprocess, batched=True, remove_columns=ds[split].column_names)

	processed_ds = DatasetDict(processed)

	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

	metric = load_metric("sacrebleu")

	def postprocess_text(preds, labels):
		preds = [pred.strip() for pred in preds]
		labels = [[label.strip()] for label in labels]
		return preds, labels

	def compute_metrics(eval_preds):
		preds, labels = eval_preds
		if isinstance(preds, tuple):
			preds = preds[0]
		preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
		preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
		labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
		preds, labels = postprocess_text(preds, labels)
		result = metric.compute(predictions=preds, references=labels)
		return {"bleu": result["score"] if "score" in result else result.get("bleu", 0.0)}

	training_args = TrainingArguments(
		output_dir=cfg.training.output_dir,
		per_device_train_batch_size=cfg.training.per_device_train_batch_size,
		per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
		gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
		learning_rate=cfg.training.learning_rate,
		weight_decay=cfg.training.weight_decay,
		num_train_epochs=cfg.training.num_train_epochs,
		warmup_ratio=cfg.training.warmup_ratio,
		logging_steps=cfg.training.logging_steps,
		evaluation_strategy=cfg.training.eval_strategy,
		eval_steps=cfg.training.eval_steps,
		save_steps=cfg.training.save_steps,
		save_total_limit=cfg.training.save_total_limit,
		fp16=cfg.training.fp16,
		predict_with_generate=True,
		auto_find_batch_size=False,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=processed_ds.get(train_split),
		eval_dataset=processed_ds.get(eval_split),
		data_collator=data_collator,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
	)

	trainer.train()
	trainer.save_model(cfg.training.output_dir)


if __name__ == "__main__":
	main()
