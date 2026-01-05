"""Training loop and evaluation logic."""

import time
import os
import logging
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from data import InfiniteDataLoader

logger = logging.getLogger(__name__)

class Trainer:
    """Manages the training lifecycle."""

    def __init__(self, model: torch.nn.Module, train_loader: InfiniteDataLoader, 
                 test_dataset, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.test_dataset = test_dataset
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(0.9, 0.99), 
            eps=1e-8
        )
        self.writer = SummaryWriter(log_dir=config.work_dir)
        self.step = 0
        self.best_loss = float('inf')

    @torch.no_grad()
    def evaluate(self, dataset, batch_size: int = 50, max_batches: int = 20) -> float:
        """Evaluates model performance on a given dataset."""
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=batch_size, num_workers=0
        )
        losses = []
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            X, Y = [t.to(self.config.device) for t in batch]
            _, loss = self.model(X, Y)
            losses.append(loss.item())
        
        self.model.train()
        return torch.tensor(losses).mean().item()

    def save_checkpoint(self, loss: float):
        """Saves the model checkpoint."""
        if loss < self.best_loss:
            self.best_loss = loss
            path = os.path.join(self.config.work_dir, "model.pt")
            logger.info(f"New best loss: {loss:.4f}. Saving model to {path}")
            torch.save(self.model.state_dict(), path)

    def train(self):
        """Main training loop."""
        self.model.train()
        logger.info("Starting training...")
        
        while True:
            t0 = time.time()
            X, Y = self.train_loader.next()
            
            # Forward & Backward
            _, loss = self.model(X, Y)
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Sync for timing if on GPU
            if self.config.device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = time.time()

            # Logging
            if self.step % 10 == 0:
                logger.info(f"Step {self.step} | Loss {loss.item():.4f} | Time {(t1-t0)*1000:.2f}ms")

            # Evaluation & Checkpointing
            if self.step > 0 and self.step % 500 == 0:
                train_loss = self.evaluate(self.train_loader.loader.dataset)
                test_loss = self.evaluate(self.test_dataset)
                
                self.writer.add_scalar("Loss/train", train_loss, self.step)
                self.writer.add_scalar("Loss/test", test_loss, self.step)
                logger.info(f"Step {self.step}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}")
                
                self.save_checkpoint(test_loss)

            self.step += 1
            if self.config.max_steps >= 0 and self.step >= self.config.max_steps:
                break