# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is an open-source foundation model for financial candlestick (K-line) forecasting, accepted at AAAI 2026. It uses a two-stage architecture: a specialized tokenizer that quantizes OHLCV data into hierarchical discrete tokens via Binary Spherical Quantization (BSQ), followed by an autoregressive Transformer that predicts future candlesticks from these tokens. Models are published to HuggingFace Hub under the `NeoQuasar` organization.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/test_kronos_regression.py
# Tests run on CPU, download models from HuggingFace Hub, and verify output consistency
# Regression tests are parameterized by context length (512, 256)
```

### Web UI
```bash
python webui/run.py          # Flask server on http://localhost:7070
```

### Fine-tuning
Two pipelines exist — Qlib-based (Chinese A-shares) and CSV-based (general data):
```bash
# Qlib fine-tuning
python finetune/train_tokenizer.py   # Train tokenizer
python finetune/train_predictor.py   # Train predictor

# CSV fine-tuning (configured via YAML)
python finetune_csv/train_sequential.py   # Sequential: tokenizer then base model
```

## Architecture

### Core Model (`model/`)

Three main classes form the API surface, all importable via `from model import ...`:

- **KronosTokenizer** (`kronos.py`) — Encoder-decoder with BSQuantizer. Encodes OHLCV into discrete hierarchical token pairs (s1 "pre" tokens, s2 "post" tokens). Methods: `encode()`, `decode()`, `indices_to_bits()`.
- **Kronos** (`kronos.py`) — Autoregressive Transformer with hierarchical token embedding, temporal embeddings, and a dual-head output (s1 + s2 predictions). Uses `DependencyAwareLayer` to condition s2 on sampled s1. Supports teacher forcing and separate `decode_s1`/`decode_s2` for two-pass inference.
- **KronosPredictor** (`kronos.py`) — High-level inference wrapper. Handles data normalization, context windowing, tokenization, autoregressive generation with top-k/top-p sampling, and denormalization back to OHLCV values. This is the primary entry point for predictions.

### Building Blocks (`model/module.py`)

Contains all reusable neural network components: `TransformerBlock`, `BSQuantizer` (Binary Spherical Quantizer), `HierarchicalEmbedding`, `TemporalEmbedding`, `DependencyAwareLayer`, `DualHead`, `RMSNorm`, and `DifferentiableEntropyFunction` (custom autograd for entropy loss).

### Token Hierarchy

The model uses a two-level token scheme:
- **s1 (pre tokens)**: Coarse quantization using `s1_bits` dimensions — captures primary price movement
- **s2 (post tokens)**: Fine quantization using `s2_bits` dimensions — captures refinement detail
- Total codebook dimension = `s1_bits + s2_bits`
- s2 decoding is conditioned on s1 via `DependencyAwareLayer`

### Fine-tuning Pipelines

- **`finetune/`** — Qlib integration for Chinese A-share markets. Config via `Config` class in `config.py`. Requires Qlib data at `~/.qlib/qlib_data/cn_data`.
- **`finetune_csv/`** — General CSV-based fine-tuning. Config via YAML loaded through `ConfigLoader`/`CustomFinetuneConfig` in `config_loader.py`. Supports sequential tokenizer-then-basemodel training.

### Web UI (`webui/`)

Flask app with `app.py` serving a candlestick prediction interface. Additional dependencies in `webui/requirements.txt` (flask, flask-cors, plotly).

### Model Zoo

Pre-trained models on HuggingFace Hub (`NeoQuasar/`):
- `Kronos-Tokenizer-base` — The tokenizer
- `Kronos-mini`, `Kronos-small`, `Kronos-base`, `Kronos-large` — Predictor variants at different scales

Models are loaded via `PyTorchModelHubMixin` from HuggingFace Hub (`from_pretrained()`).

## Key Conventions

- All model classes use `PyTorchModelHubMixin` for HuggingFace Hub integration
- Input data columns: `open`, `high`, `low`, `close`, `volume`, `amount`
- Maximum context length: 512 tokens
- Tests pin specific model revisions (`MODEL_REVISION`, `TOKENIZER_REVISION`) for reproducibility
- The `sys.path.append("../")` pattern is used in some modules for imports — be aware when running scripts from different directories
