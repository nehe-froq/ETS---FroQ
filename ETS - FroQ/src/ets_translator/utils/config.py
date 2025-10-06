from __future__ import annotations

import pathlib
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
	path: str
	name: Optional[str] = None
	split_train: str = Field(default="train")
	split_eval: str = Field(default="validation")
	text_column: str = Field(default="translation")


class TrainingConfig(BaseModel):
	output_dir: str = Field(default="outputs/run")
	per_device_train_batch_size: int = 8
	per_device_eval_batch_size: int = 8
	gradient_accumulation_steps: int = 1
	learning_rate: float = 5e-5
	weight_decay: float = 0.0
	num_train_epochs: float = 1.0
	warmup_ratio: float = 0.0
	logging_steps: int = 50
	eval_strategy: str = Field(default="epoch")
	eval_steps: Optional[int] = None
	save_steps: Optional[int] = None
	save_total_limit: int = 2
	fp16: bool = False


class TranslationConfig(BaseModel):
	experiment_name: str = Field(default="exp")
	seed: int = 42
	model_name: str
	source_lang: str
	target_lang: str
	max_source_length: int = 256
	max_target_length: int = 256

	hf_dataset: DatasetConfig
	training: TrainingConfig

	@staticmethod
	def load(path: str | pathlib.Path) -> "TranslationConfig":
		path = pathlib.Path(path)
		with path.open("r") as f:
			data = yaml.safe_load(f)
		return TranslationConfig(**data)
