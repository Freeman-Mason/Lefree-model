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
