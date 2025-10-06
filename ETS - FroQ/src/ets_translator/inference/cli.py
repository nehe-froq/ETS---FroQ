from __future__ import annotations

import pathlib
from typing import Optional

import typer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = typer.Typer(add_completion=False)


def _load(model_path: str):
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
	return tokenizer, model


@app.command()
def main(
	text: str = typer.Argument(..., help="Source text to translate"),
	model: str = typer.Option("Helsinki-NLP/opus-mt-en-de", help="Model or path"),
	max_new_tokens: int = typer.Option(128, help="Max new tokens"),
	temperature: float = typer.Option(1.0, help="Sampling temperature"),
	top_p: float = typer.Option(1.0, help="Top-p nucleus sampling"),
	top_k: int = typer.Option(50, help="Top-k sampling"),
):
	tokenizer, model = _load(model)
	inputs = tokenizer([text], return_tensors="pt")
	outputs = model.generate(
		**inputs,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
		do_sample=temperature != 1.0 or top_p < 1.0 or top_k < 50,
	)
	print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


if __name__ == "__main__":
	app()
