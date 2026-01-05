"""Inference script for generating text from trained models."""

import argparse
import logging
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional

from config import ModelConfig
from models import Transformer
from data import create_datasets

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Makemore Inference")
    
    # System
    parser.add_argument('--model-path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--input-file', type=str, default='names.txt', help="Original input file to rebuild vocab")
    parser.add_argument('--device', type=str, default='cpu')
    
    # Generation
    parser.add_argument('--num-samples', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--max-new-tokens', type=int, default=20, help="Max tokens to generate per sample")
    parser.add_argument('--temperature', type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument('--top-k', type=int, default=200, help="Top-k filtering")
    parser.add_argument('--prompt', type=str, default='', help="Starting string prompt")
    
    # Model Architecture (Must match training!)
    parser.add_argument('--type', type=str, default='transformer')
    parser.add_argument('--n-layer', type=int, default=4)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-embd', type=int, default=64)

    return parser.parse_args()

@torch.no_grad()
def generate(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, 
             temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    """
    Generates new tokens using the trained model.

    Args:
        model: The trained PyTorch model.
        idx: LongTensor of shape (b, t) containing the current context.
        max_new_tokens: Number of tokens to generate.
        temperature: Factor to scale logits (1.0 = standard, >1.0 = chaotic, <1.0 = confident).
        top_k: If set, only sample from the top k most likely tokens.

    Returns:
        Tensor containing the original context plus generated tokens.
    """
    block_size = model.get_block_size()
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Break if we hit the special end token (index 0 usually reserved for padding/end in this setup)
        if idx_next.item() == 0:
            break

    return idx

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Rebuild Dataset/Vocab
    # In a full production env, we would load a vocab.json artifact.
    # Here, we rebuild it from source to ensure consistency with data.py.
    logger.info("Rebuilding vocabulary from input file...")
    try:
        train_ds, _ = create_datasets(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    vocab_size = train_ds.get_vocab_size()
    block_size = train_ds.get_output_length()
    
    # 2. Initialize Model
    logger.info(f"Initializing {args.type} model structure...")
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    
    if args.type == 'transformer':
        model = Transformer(config)
    else:
        raise NotImplementedError(f"Model type {args.type} not supported in inference.")
    
    # 3. Load Checkpoint
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    logger.info(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Prepare Context
    # If prompt is empty, start with the specific start token (usually 0 or context padding)
    # Based on data.py, 0 is the special token.
    if args.prompt:
        start_ids = train_ds.encode(args.prompt).unsqueeze(0).to(device)
    else:
        # Start with a standard zero token (padding/start) 
        start_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 5. Generate
    logger.info(f"Generating {args.num_samples} samples...")
    print("-" * 50)
    
    for i in range(args.num_samples):
        y = generate(
            model, 
            start_ids, 
            max_new_tokens=args.max_new_tokens, 
            temperature=args.temperature, 
            top_k=args.top_k
        )
        
        # Decode and Print
        # We skip the first token if it was just the initialization zero
        row = y[0].tolist()
        decoded = train_ds.decode(row)
        print(f"{i+1}: {decoded}")

    print("-" * 50)

if __name__ == '__main__':
    main()