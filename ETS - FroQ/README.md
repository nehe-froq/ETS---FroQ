# ETS Translator

End-to-end translation project with training and inference using Hugging Face Transformers.

## Features
- Train or fine-tune seq2seq translation models (e.g., Helsinki-NLP/Opus-MT, T5)
- Config-driven (YAML) setup
- Evaluation with BLEU (sacreBLEU)
- Inference via CLI and FastAPI server

## Quickstart

1) Create venv and install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Configure `configs/translation.yaml`:
```yaml
experiment_name: demo-mt
seed: 42
model_name: Helsinki-NLP/opus-mt-en-de
source_lang: en
target_lang: de
max_source_length: 256
max_target_length: 256

hf_dataset:
  path: wmt14
  name: de-en
  split_train: train
  split_eval: test
  text_column: translation

training:
  output_dir: outputs/demo-mt
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  weight_decay: 0.0
  num_train_epochs: 1
  warmup_ratio: 0.03
  logging_steps: 50
  eval_strategy: steps
  eval_steps: 200
  save_steps: 200
  save_total_limit: 2
  fp16: false
```

3) Train:
```bash
python -m ets_translator.training.train --config configs/translation.yaml
```

4) CLI inference:
```bash
ets-translate --model outputs/demo-mt "Hello world!"
```

5) API server:
```bash
python -m ets_translator.inference.server --model outputs/demo-mt --host 0.0.0.0 --port 8000
# then POST {"text": "Hello world!"} to /translate
```

## Notes for Apple Silicon
- To use MPS acceleration, install torch following the instructions at `https://pytorch.org`.
- CPU-only is supported but slower.

## License
MIT
