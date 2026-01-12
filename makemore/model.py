"""Neural network model definitions for the Makemore project.

This module contains various autoregressive character-level language models,
ranging from simple Bigram models to complex Transformers and RNNs.

Typical usage example:

    config = ModelConfig(vocab_size=65, block_size=8)
    model = Transformer(config)
    logits, loss = model(input_ids, targets)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig

# -----------------------------------------------------------------------------
# Helper Modules
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """Implementation of the Gaussian Error Linear Units (GELU) activation function.
    
    This version approximates the GELU function as described in the BERT paper.
    Reference: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with GELU activation applied.
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# -----------------------------------------------------------------------------
# Transformer Components
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention layer.

    This layer implements scaled dot-product attention with a causal mask to
    ensure the model cannot attend to future tokens.
    """

    def __init__(self, config: ModelConfig):
        """Initializes the CausalSelfAttention layer.

        Args:
            config: Model configuration object containing 'n_embd', 'n_head', 
                    and 'block_size'.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "Embedding dimension must be divisible by number of heads."
        
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Causal mask (buffer)
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            x: Input tensor of shape (Batch, Time, Channels).

        Returns:
            Output tensor of shape (Batch, Time, Channels).
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # (B, nh, T, hs) where nh = n_head and hs = head_size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Aggregate values
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """A standard Transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer Language Model.
    
    A decoder-only Transformer architecture suitable for autoregressive tasks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init weights (optional but recommended for convergence)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes logits and loss.

        Args:
            idx: Input token indices of shape (Batch, Time).
            targets: Target token indices of shape (Batch, Time).

        Returns:
            Tuple containing:
                logits: Tensor of shape (Batch, Time, Vocab Size).
                loss: Scalar Tensor (if targets provided) else None.
        """
        device = idx.device
        b, t = idx.size()
        
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.block_size}")
            
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# -----------------------------------------------------------------------------
# Bag of Words (BoW) Components
# -----------------------------------------------------------------------------

class CausalBoW(nn.Module):
    """Causal Bag of Words layer.
    
    Averages the preceding element vectors. It resembles a CausalAttention module
    but uses fixed averaging weights instead of learned attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Weighted average of all preceding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)
        return y


class BoWBlock(nn.Module):
    """Collects BoW features and passes them through an MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, config.n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cbow(x)
        x = x + self.mlp(x)
        return x


class BoW(nn.Module):
    """Bag of Words Language Model.
    
    Encodes previous tokens and positions, averages them, and predicts the next token.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.context_block = BoWBlock(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.block_size}")
            
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        
        x = tok_emb + pos_emb
        x = self.context_block(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# -----------------------------------------------------------------------------
# RNN / GRU Components
# -----------------------------------------------------------------------------

class RNNCell(nn.Module):
    """Vanilla RNN Cell."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Input (n_embd) + Hidden (n_embd2) -> New Hidden (n_embd2)
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt: torch.Tensor, hprev: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([xt, hprev], dim=1)
        ht = torch.tanh(self.xh_to_h(xh))
        return ht


class GRUCell(nn.Module):
    """Gated Recurrent Unit (GRU) Cell."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        input_dim = config.n_embd + config.n_embd2
        # Gate weights
        self.xh_to_z = nn.Linear(input_dim, config.n_embd2)
        self.xh_to_r = nn.Linear(input_dim, config.n_embd2)
        self.xh_to_hbar = nn.Linear(input_dim, config.n_embd2)

    def forward(self, xt: torch.Tensor, hprev: torch.Tensor) -> torch.Tensor:
        # 
        xh = torch.cat([xt, hprev], dim=1)
        
        # Reset gate
        r = torch.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        
        # Candidate hidden state
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = torch.tanh(self.xh_to_hbar(xhr))
        
        # Update gate
        z = torch.sigmoid(self.xh_to_z(xh))
        
        # Final state
        ht = (1 - z) * hprev + z * hbar
        return ht


class RNN(nn.Module):
    """Recurrent Neural Network Language Model (Supports Vanilla RNN and GRU)."""

    def __init__(self, config: ModelConfig, cell_type: str = 'rnn'):
        """Initializes the RNN.

        Args:
            config: Model configuration.
            cell_type: Type of cell to use ('rnn' or 'gru').
        """
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # Starting hidden state
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
            
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()

        emb = self.wte(idx) # (b, t, n_embd)

        hprev = self.start.expand((b, -1)) # Expand batch dimension
        hiddens = []
        
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# -----------------------------------------------------------------------------
# Baseline Models
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-Layer Perceptron Language Model.
    
    Encodes the previous block_size tokens, concatenates them, and predicts the next token.
    Reference: Bengio et al. 2003
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # +1 for special <BLANK> token used for padding start of sequence
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Gather embeddings for the previous block_size words
        # Note: This logic shifts inputs to create context windows
        embs = []
        curr_idx = idx
        
        for _ in range(self.block_size):
            tok_emb = self.wte(curr_idx)
            # Create a shifted version for the next context position
            curr_idx = torch.roll(curr_idx, 1, 1)
            # Fill the rolled position with the special blank token (stored at vocab_size)
            curr_idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        # Concatenate all embeddings: (b, t, n_embd * block_size)
        x = torch.cat(embs, -1) 
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


class Bigram(nn.Module):
    """Bigram Language Model.
    
    A simple lookup table where logits for the next character depend solely 
    on the current character.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self) -> int:
        return 1

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Simple lookup
        logits = self.logits[idx]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss