# gpt-from-scratch
GPT from scratch using a custom Transformer model with RoPE implementation, markdown explanations, and ablation experiments

## Overview
This project builds a GPT Transformer model trained on Shakespeare from scratch, implements RoPE (Rotary Positional Embeddings) from the original paper, documents the underlying mechanisms of each component, and performs ablation experiments on certain features. In this case, I removed positional embeddings, residual connections, and head scaling factor, running them one at a time. Without positional embeddings, the ablative model still performed similarly to the regular one. Due to its small size, it can memorize local patterns and pick up cues from token statistics. Without residual connections, the model performed dramatically worse than usual since gradients now had to pass through numerous nonlinear layers that destroyed features and overwrote representations. Without a scaling factor, the model once again performed similarly to its regular state. This is because of the small head size (4), meaning the scaling factor only divided the weights by two and was therefore too tiny to have any meaningful effect. Additionally, the model was run twice using demo hyperparameters: once without RoPE and once with. RoPE drastically decreased the validation loss (1.52 vs. 1.91). These findings suggest that transformer components matter much more when the model or sequences get larger, so it's meaningful to study the role of each component and their sizes on a larger scale.

## Results
Demo Comparison (n_embd=128, n_head=4, n_layer=4, block_size=256, max_iters=5000)
| Model | Train Loss | Validation Loss |
|-------|------------|-----------------|
| Base GPT | 1.7430 | 1.9113 |
| RoPE GPT | 1.2892 | 1.5232 |

Ablation Comparison (n_embd=16, n_head=4, n_layer=4, block_size=8, max_iters=3000)
| Ablation | Train Loss | Validation Loss |
|----------|------------|-----------------|
| None | 2.2788 | 2.2906 |
| Positional Embedding | 2.2764 | 2.2869 |
| Residual Connections | 3.1892 | 3.1796 |
| Scaling Factor | 2.2640 | 2.2766 |

## Key Findings
The ablation experiments and RoPE comparison revealed that Transformer components matter much more when the model or sequences get larger. The scaling factor ablation showed little effect at a small scale, but at a large scale it would matter significantly because larger head sizes mean larger dot products. The positional embedding ablation also showed little effect at a small scale, but at a large scale the model is forced to learn without picking up patterns or cues. Therefore, it's meaningful to study the role of each component and their sizes on a larger scale.

## Notebook Structure
The notebook starts off with importing the libraries, reading the dataset, defining the demo hyperparameters used for the Base vs. RoPE comparison, and initializing the batch and loss functions for the data. It is then split into nine sections, with a markdown cell at the top of each section and the corresponding code below it. They go in this order: naive averaging, matrix multiplication, masked softmax, single-head attention in raw NumPy, single-head attention in PyTorch, multi-head attention, full GPT, RoPE, and ablations.

## Requirements
- Python 3.8+
- numpy
- torch

## Usage
In notebook, run the following command to import the dataset:

import urllib.request

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input.txt"
)
