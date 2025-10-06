from __future__ import annotations

from typing import Dict, Any

from datasets import load_dataset


def load_translation_dataset(cfg: Dict[str, Any]):
	return load_dataset(
		path=cfg["path"],
		name=cfg.get("name"),
		split=None,
	)


def build_preprocess_functions(tokenizer, source_lang: str, target_lang: str, max_source_length: int, max_target_length: int):
	prefix_src = source_lang
	prefix_tgt = target_lang

	def preprocess(batch):
		translations = batch["translation"]
		src_texts = [ex[prefix_src] for ex in translations]
		tgt_texts = [ex[prefix_tgt] for ex in translations]

		model_inputs = tokenizer(
			src_texts,
			max_length=max_source_length,
			truncation=True,
		)

		with tokenizer.as_target_tokenizer():
			labels = tokenizer(
				tgt_texts,
				max_length=max_target_length,
				truncation=True,
			)

		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	return preprocess
