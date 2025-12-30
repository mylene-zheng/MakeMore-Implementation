# makemore: Deep Learning Implementation Study

> **Note:** This repository represents my implementation of the `makemore` project, completed as an independent study alongside my **L3 University curriculum**. It serves as a comprehensive exploration of autoregressive character-level language models.

## Project Overview

`makemore` takes a text file as input (where each line is assumed to be one training example) and generates new, similar data. Under the hood, it is an autoregressive character-level language model.

This project implements a wide trajectory of architectures, starting from simple statistics and scaling up to modern deep learning techniques used in GPT. For example, feeding it a database of names allows it to generate unique, name-like sequences.

The codebase is designed to be "hackable" and lightweight, using **PyTorch** as the only major dependency.

## Implemented Architectures

Through this project, I have implemented and trained the following models, tracing the history of sequence modeling papers:

* **Bigram Model:** A baseline approach where one character predicts the next using a lookup table of counts.
* **MLP (Multi-Layer Perceptron):** A neural language model following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
* **CNN:** A convolutional approach following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499).
* **RNN (Recurrent Neural Network):** Following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf).
* **LSTM (Long Short-Term Memory):** Following [Graves et al. 2014](https://arxiv.org/abs/1308.0850).
* **GRU (Gated Recurrent Unit):** Following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259).
* **Transformer:** A self-attention based architecture following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762).

## Usage

The repository includes a `names.txt` dataset (32K common names from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018).

**Training**
To train a model (default is a small Transformer):
```bash
$ python makemore.py -i names.txt -o names
```

* Training progress, logs, and model checkpoints are saved to the `names` directory.

* This implementation runs efficiently on standard hardware (CPU) but accelerates significantly with a GPU.

**Sampling** To generate samples from a trained model:
```bash
$ python makemore.py -i names.txt -o names --sample-only
```
**Example Output**
Generated names (test logprob ~1.92):

dontell

khylum

camatena

aeriline

...

lucianno






