
# Makemore: Production-Grade Autoregressive Language Model

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-google-blueviolet)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

A robust, modular, and type-safe implementation of autoregressive character-level language models. This project depends on the original original concept of [makemore](https://github.com/karpathy/makemore) educational script into a scalable Python package structure, adhering to **Google Python Style Guides** and modern MLOps standards.

The primary objective of this repository is to serve as a reference implementation for structuring PyTorch research code for production readiness, featuring separation of concerns, strict configuration management, and comprehensive logging.



## 📖 Table of Contents

- [Makemore: Production-Grade Autoregressive Language Model](#makemore-production-grade-autoregressive-language-model)
  - [📖 Table of Contents](#-table-of-contents)
  - [🧐 Overview](#-overview)
  - [✨ Key Features](#-key-features)
  - [🏗 Project Architecture](#-project-architecture)
    - [Model Architecture](#model-architecture)
  - [💻 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [🚀 Usage](#-usage)
    - [Training](#training)
    - [Inference](#inference)
  - [⚙ Configuration](#-configuration)
    - [Model Arguments](#model-arguments)
    - [Training Arguments](#training-arguments)
  - [🧮 Mathematical Context](#-mathematical-context)
  - [🛠 Development](#-development)
    - [Code Style](#code-style)
    - [Running Tests](#running-tests)
  - [📄 License](#-license)
  - [🙏 Acknowledgements](#-acknowledgements)

---

## 🧐 Overview

"Makemore" treats language generation as a next-token prediction task. Given a dataset of text (e.g., a list of names), the model is trained to maximize the likelihood of the next character in a sequence given the history of previous characters.

$$P(x) = \prod_{t=1}^{T} P(x_t \mid x_{\lt t})$$

While the core logic remains faithful to the original educational intent, the infrastructure has been rebuilt to support:
* **Scalability:** Easy to extend to larger datasets and deeper models.
* **Reproducibility:** Deterministic seeding and config serialization.
* **Observability:** Integrated TensorBoard logging for loss tracking.

---

## ✨ Key Features

* **Modular Design:** Decoupled logic for Data (`data.py`), Modeling (`models.py`), Training (`trainer.py`), and Configuration (`config.py`).
* **Strict Typing:** Utilization of Python's `typing` module and `dataclasses` to ensure code reliability and IDE support.
* **Google Style Guide:** Docstrings and code formatting adhere to the Google Python Style Guide.
* **Modern Transformer:** Implementation of a decoder-only Transformer with `LayerNorm` and `GELU` activations.
* **Robust Data Pipeline:** Custom `InfiniteDataLoader` leveraging Python generators for seamless streaming of batches.

---

## 🏗 Project Architecture

The codebase is organized to separate configuration, data ingestion, and modeling logic.

```text
makemore/
├── config.py           # Dataclasses for Model and Training configuration
├── data.py             # Dataset classes and InfiniteDataLoader
├── models.py           # PyTorch definitions (Transformer, CausalSelfAttention)
├── trainer.py          # Training loop, checkpointing, and evaluation engine
├── main.py             # CLI entry point for training
├── inference.py        # CLI entry point for generation/sampling
└── names.txt           # Default training corpus

Jupyter-Notes/          # Code Snippets and Explanations Provided as Jypyter Notebook files 
├── names.txt           # Default training corpus
├── note1.ipynb             
├── note2.ipynb         
├── note3.ipynb         
├── note4.ipynb         
├── note5.ipynb         
└── note6.ipynb         
```

### Model Architecture

The default model is a Decoder-only Transformer. It utilizes causal masking to ensure that predictions for position  depend only on positions .

---

## 💻 Installation

### Prerequisites

* Python 3.8 or higher
* PyTorch 2.0 or higher (CUDA recommended for training)

### Setup

1. **Clone the repository**
```bash
git clone [https://github.com/your-org/makemore-refactored.git](https://github.com/your-org/makemore-refactored.git)
cd makemore-refactored

```


2. **Create a virtual environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install dependencies**
```bash
pip install torch torchvision torchaudio tensorboard

```



---

## 🚀 Usage

### Training

To start training the model, use `main.py`. The script will automatically split the data, initialize the model, and begin the optimization loop.

**Basic CPU Run:**

```bash
python main.py --input-file names.txt --work-dir out/run_01

```

**Advanced GPU Run:**

```bash
python main.py \
    --input-file names.txt \
    --work-dir out/transformer_gpu \
    --device cuda \
    --type transformer \
    --n-layer 6 \
    --n-head 6 \
    --n-embd 128 \
    --batch-size 64 \
    --max-steps 5000 \
    --learning-rate 0.001

```

**Monitoring Progress:**
Logs are written to the `work-dir`. View them using TensorBoard:

```bash
tensorboard --logdir out/

```

### Inference

Generate new samples using a trained checkpoint. Ensure the model architecture arguments match those used during training.

```bash
python inference.py \
    --model-path out/transformer_gpu/model.pt \
    --input-file names.txt \
    --device cuda \
    --n-layer 6 \
    --n-head 6 \
    --n-embd 128 \
    --num-samples 10 \
    --temperature 0.8

```

---

## ⚙ Configuration

The project uses `argparse` mapped to strict `dataclasses` in `config.py`.

### Model Arguments

| Argument    | Default       | Description                                           |
| ----------- | ------------- | ----------------------------------------------------- |
| `--type`    | `transformer` | Architecture type (currently supports `transformer`). |
| `--n-layer` | `4`           | Number of transformer blocks.                         |
| `--n-head`  | `4`           | Number of multi-head attention heads.                 |
| `--n-embd`  | `64`          | Dimensionality of the embeddings.                     |

### Training Arguments

| Argument          | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `--batch-size`    | `32`    | Number of sequences per training batch.       |
| `--learning-rate` | `5e-4`  | Peak learning rate for AdamW optimizer.       |
| `--max-steps`     | `-1`    | Max optimization steps. `-1` runs infinitely. |
| `--seed`          | `3407`  | Random seed for reproducibility.              |

---

## 🧮 Mathematical Context

The model optimizes the **Negative Log Likelihood (NLL)** loss.

For a sequence of characters , the model estimates the joint probability:
$$P(x) = \prod_{i=1}^{T} P(x_i \mid x_1, \dots, x_{i-1})$$

During training, we minimize the cross-entropy loss between the predicted distribution of the next character and the actual next character in the training data.

The **Self-Attention** mechanism allows the model to weigh the importance of different previous characters dynamically:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
where a causal mask ensures  is lower-triangular, preventing information leakage from the future.

---

## 🛠 Development

### Code Style

This project enforces the **Google Python Style Guide**.

* **Docstrings:** All modules, classes, and public methods must have descriptive docstrings.
* **Type Hints:** Use `typing` for all function signatures.

### Running Tests

(Placeholder for testing instructions)

```bash
python -m unittest discover tests

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🙏 Acknowledgements

* Original concept **Andrej Karpathy** ([makemore](https://github.com/karpathy/makemore)).
