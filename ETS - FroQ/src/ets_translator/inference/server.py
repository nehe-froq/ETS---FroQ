from __future__ import annotations

import argparse
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import uvicorn

app = FastAPI()

_tokenizer = None
_model = None


class TranslateRequest(BaseModel):
	text: str
	max_new_tokens: int = 128
	temperature: float = 1.0
	top_p: float = 1.0
	top_k: int = 50


class TranslateResponse(BaseModel):
	translation: str


@app.on_event("startup")
async def _startup_event():
	pass


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
	inputs = _tokenizer([req.text], return_tensors="pt")
	outputs = _model.generate(
		**inputs,
		max_new_tokens=req.max_new_tokens,
		temperature=req.temperature,
		top_p=req.top_p,
		top_k=req.top_k,
		do_sample=req.temperature != 1.0 or req.top_p < 1.0 or req.top_k < 50,
	)
	text = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
	return TranslateResponse(translation=text)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="Helsinki-NLP/opus-mt-en-de")
	parser.add_argument("--host", type=str, default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()

	global _tokenizer, _model
	_tokenizer = AutoTokenizer.from_pretrained(args.model)
	_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

	uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
	main()
