---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - agriculture
  - causal-lm
  - fine-tuned
---

# Train Model

Fine-tunes a causal language model on local data.

## Data

By default, the script reads files from the `data/` folder.

Supported formats:

- `.xlsx` / `.xls`
- `.csv`
- `.json` / `.jsonl`
- `.txt`

For spreadsheet data, each row is turned into a text record before training.

You can also load the Hugging Face crop dataset:

```powershell
python import_hf_dataset.py
```

## Install

```powershell
pip install -r requirements.txt
```

## Model

Change the model in `.env`:

```text
MODEL_NAME=Qwen/Qwen3-0.6B
```

## Push To Hugging Face

Set these in `.env`:

```text
HF_TOKEN=your_huggingface_token
HF_REPO_ID=your-username/your-model-name
PUSH_TO_HUB=true
HF_PRIVATE_REPO=false
```

Then run:

```powershell
python main.py
```

Or push only when you choose:

```powershell
python main.py --push_to_hub --hub_model_id your-username/your-model-name
```

## Run

```powershell
python main.py
```

This uses:

- model: value from `.env`
- data folder: `data`
- output folder: `Lefree-01-model`

## Common Options

```powershell
python main.py --epochs 3 --output_dir agriculture-model
```

```powershell
python main.py --data_file data --max_length 512
```

You can still override the model from the command line:

```powershell
python main.py --model_name Qwen/Qwen3-0.6B
```

## Evaluate South Africa Agri Intelligence

The repo includes a benchmark for South Africa-specific agriculture advice:

```powershell
python evaluate_sa_agri.py --model_dir Lefree-01-model
```

Run training first so `Lefree-01-model` contains saved model and tokenizer files:

```powershell
python main.py
```

You can also evaluate a Hugging Face model directly:

```powershell
python evaluate_sa_agri.py --model_dir Qwen/Qwen3-0.6B
```

This asks the model questions about maize diseases, Free State planting windows,
soil pH, dryland rainfall, liming, stem borers, weed control, fertilisation and
responsible uncertainty.

The safety and honesty rules used for advisory behavior are documented in:

```text
docs/safety_and_honesty.md
```

Detailed results are written to:

```text
eval_results/sa_agri_results.jsonl
```

Use the score as a progress signal:

- `0.80+`: strong enough for the current benchmark
- `0.60-0.79`: useful but still missing important local concepts
- `<0.60`: not ready for South Africa agri advisory use

The benchmark questions and reference answers live in:

```text
benchmarks/south_africa_agri_eval.jsonl
```
